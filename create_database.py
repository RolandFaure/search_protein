import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, T5Tokenizer, BertConfig
import numpy as np
import time
import sys
from sklearn.metrics.pairwise import cosine_similarity
import random
import argparse
import os
import fcntl
import faiss
import time
import multiprocessing
from usearch.index import Index
from multiprocessing import Pool, Manager
from functools import partial
import shutil


d = 512

__version__ = "2.1.0"

class UncenteredPCA:
    """
    Uncentered PCA using SVD-based rotation without mean subtraction.
    This preserves cosine distances between normalized vectors better than centered PCA.
    """
    def __init__(self, n_components):
        self.n_components_ = n_components
        self.components_ = None  # Shape: (n_components, d)
        self.singular_values_ = None
        self.mean_ = None  # Always zero for uncentered PCA
    
    def fit(self, X):
        """Fit uncentered PCA using SVD on the data directly (no centering)."""
        # Compute SVD without centering
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        
        # Store the rotation matrix from right singular vectors
        self.components_ = Vt[:self.n_components_, :]  # Shape: (n_components, d)
        self.singular_values_ = S[:self.n_components_]
        self.mean_ = np.zeros(X.shape[1], dtype=np.float32)  # Zero mean for uncentered
        return self
    
    def transform(self, X):
        """Apply rotation transformation: X @ components_.T"""
        return X @ self.components_.T


def compute_and_save_pca(embedding_file, database_folder, n_components=64, n_samples=10_000_000):
    """
    Compute uncentered PCA using SVD and save the model.
    
    Uncentered PCA preserves cosine distances better than standard (centered) PCA.
    
    Args:
        embedding_file (str): Path to the embeddings file.
        database_folder (str): Path to folder where PCA model and stats will be saved.
        n_components (int): Number of PCA components (default: 64).
        n_samples (int): Number of vectors to use for PCA fitting (default: 10_000_000).
    
    Returns:
        UncenteredPCA model object
    """
    print(f"Computing uncentered PCA with {n_components} components using {n_samples} samples...")
    pca_folder = os.path.join(database_folder, "pca")
    os.makedirs(pca_folder, exist_ok=True)
    
    bytes_per_vector = d * 2  # Each float16 is 2 bytes
    
    # Load first n_samples vectors
    print(f"Loading first {n_samples} vectors from embeddings...")
    with open(embedding_file, "rb") as ef:
        bytes_to_read = min(n_samples * bytes_per_vector, os.path.getsize(embedding_file))
        bytes_data = ef.read(bytes_to_read)
        num_vectors = len(bytes_data) // bytes_per_vector
        vectors = np.frombuffer(bytes_data, dtype=np.float16).reshape(-1, d).astype(np.float32)
    
    print(f"Loaded {num_vectors} vectors for PCA fitting")
    
    # Fit uncentered PCA using SVD
    print(f"Fitting uncentered PCA (SVD-based)...")
    pca = UncenteredPCA(n_components=n_components)
    pca.fit(vectors)
    
    # Save PCA model components as numpy files (not as pickle for portability)
    components_file = os.path.join(pca_folder, "components.npy")
    singular_values_file = os.path.join(pca_folder, "singular_values.npy")
    np.save(components_file, pca.components_)
    np.save(singular_values_file, pca.singular_values_)
    print(f"Uncentered PCA components saved to {components_file}")
    
    # Save stats
    explained_variance_ratio = (pca.singular_values_ ** 2) / np.sum(pca.singular_values_ ** 2)
    stats = {
        "n_components": n_components,
        "n_samples_used": num_vectors,
        "pca_type": "uncentered",
        "explained_variance_ratio": explained_variance_ratio.tolist(),
        "cumulative_explained_variance": np.cumsum(explained_variance_ratio).tolist(),
        "total_explained_variance": float(np.sum(explained_variance_ratio)),
        "singular_values": pca.singular_values_.tolist(),
    }
    
    import json
    stats_file = os.path.join(pca_folder, "pca_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"PCA stats saved to {stats_file}")
    print(f"Explained variance ratio: {stats['total_explained_variance']:.4f}")
    print(f"Cumulative explained variance: {stats['cumulative_explained_variance'][-1]:.4f}")
    
    return pca

def load_pca(pca_folder):
    """
    Load PCA model from folder if it exists.
    
    Args:
        pca_folder (str): Path to the PCA folder.
    
    Returns:
        tuple: (PCA model, n_components) or (None, None) if PCA doesn't exist
    """
    components_file = os.path.join(pca_folder, "components.npy")
    singular_values_file = os.path.join(pca_folder, "singular_values.npy")
    
    if not os.path.exists(components_file) or not os.path.exists(singular_values_file):
        return None, None
    
    # Load components and singular values, reconstruct UncenteredPCA object
    components = np.load(components_file)
    singular_values = np.load(singular_values_file)
    
    n_components = components.shape[0]
    pca = UncenteredPCA(n_components=n_components)
    pca.components_ = components
    pca.singular_values_ = singular_values
    pca.mean_ = np.zeros(components.shape[1], dtype=np.float32)
    
    return pca, n_components

def apply_pca_to_embeddings(embedding_file, database_folder, pca, n_components):
    """
    Transform all embeddings using PCA and save them.
    
    Args:
        embedding_file (str): Path to original embeddings file.
        database_folder (str): Path to database folder.
        pca: Fitted PCA model.
        n_components (int): Number of PCA components.
    
    Returns:
        str: Path to the transformed embeddings file
    """
    print(f"Applying PCA transformation to all embeddings...")
    
    bytes_per_vector_original = d * 2  # Original: float16
    bytes_per_vector_new = n_components * 2  # New: float16
    
    transformed_file = os.path.join(database_folder, f"{os.path.basename(embedding_file)}.pca")
    
    # Process file in chunks to manage memory
    chunk_size = 100000  # vectors per chunk
    
    with open(embedding_file, "rb") as ef:
        ef.seek(0, os.SEEK_END)
        file_size = ef.tell()
        total_vectors = file_size // bytes_per_vector_original
        ef.seek(0)
        
        with open(transformed_file, "wb") as tf:
            processed = 0
            while processed < total_vectors:
                # Read chunk
                to_read = min(chunk_size, total_vectors - processed)
                bytes_data = ef.read(to_read * bytes_per_vector_original)
                
                # Convert and transform
                chunk = np.frombuffer(bytes_data, dtype=np.float16).reshape(-1, d).astype(np.float32)
                transformed_chunk = pca.transform(chunk).astype(np.float32)
                # Normalize after PCA to preserve unit length property
                norms = np.linalg.norm(transformed_chunk, axis=1, keepdims=True)
                transformed_chunk = transformed_chunk / (norms + 1e-10)
                transformed_chunk = transformed_chunk.astype(np.float16)
                
                # Write transformed chunk
                tf.write(transformed_chunk.tobytes())
                
                processed += to_read
                if processed % (chunk_size * 10) == 0:
                    print(f"Processed {processed}/{total_vectors} vectors")
    
    print(f"PCA-transformed embeddings saved to {transformed_file}")
    return transformed_file

def initialize_model_and_tokenizer(device_id):
    """
    Initializes a model and tokenizer for natural language processing tasks.
    This function loads a pre-trained model and tokenizer from the 'tattabio/gLM2_650M_embed' 
    repository. The model is loaded with bfloat16 precision and moved to the specified device 
    (GPU or CPU). The tokenizer is also initialized for use with the same model.
    Args:
        device_id (int): The ID of the GPU device to use. If CUDA is not available, 
                         the model will be loaded on the CPU.
    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The pre-trained model loaded on the specified device.
            - tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with the model.
    """
    device = torch.device(f'cuda:{device_id}' if (torch.cuda.is_available() and device_id != 'cpu') else 'cpu')
    
    model = AutoModel.from_pretrained('tattabio/gLM2_650M_embed', revision="1b5c96057abf48f85e460a5f9a69deadc820f51c", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M_embed', revision="1b5c96057abf48f85e460a5f9a69deadc820f51c", trust_remote_code=True)
    
    return model, tokenizer

def embed_glm2_parallel(sequences_device_id):
    """
    Generates embeddings for a list of sequences in parallel using a specified device.
    Args:
        sequences_device_id (tuple): A tuple containing:
            - sequences (list of str): A list of input sequences to embed.
            - device_id (int or str): The ID of the GPU device to use for embedding, or "cpu" for CPU embedding.
    Returns:
        np.ndarray: A 2D NumPy array containing the embeddings for the input sequences.
                    Each row corresponds to the embedding of a sequence.
    Notes:
        - Sequences are processed in batches to optimize VRAM usage.
        - Embeddings are periodically transferred from GPU to CPU to manage GPU memory.
        - The function assumes that the model's output has an attribute `pooler_output`
          which contains the desired embeddings.
        - The variable `d` should be defined globally or within the model initialization
          to represent the dimensionality of the embeddings.
    """
    sequences, device_id = sequences_device_id
    start_time = time.time()
    model, tokenizer = initialize_model_and_tokenizer(device_id)
    
    for i in range(len(sequences)):
        sequences[i] = "<+>" + sequences[i].rstrip('*')

    device = torch.device(f'cuda:{device_id}' if (torch.cuda.is_available() and device_id != "cpu") else "cpu")
    embeddings_array = np.empty((0, d), dtype=np.float32)
    embeddings = torch.empty(0, d, device=device)
    batch_size = 50

    end_time = time.time()
    embedding_time_start = time.time()

    for seq_start in range(0, len(sequences), batch_size):
        encodings = tokenizer(sequences[seq_start:min(seq_start + batch_size, len(sequences))], return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            attention_mask = encodings.attention_mask.bool().to(device) #this is very important to handle the padding correctly
            pooled_embeds = model(encodings.input_ids.to(device), attention_mask=attention_mask).pooler_output
        # print("Embedded ", min(seq_start + batch_size, len(sequences)), " sequences")

        embeddings = torch.cat((embeddings, pooled_embeds), dim=0)

        if (seq_start / batch_size) % 20 == 0:
            embeddings = embeddings.float().cpu().detach().numpy()
            faiss.normalize_L2(embeddings)
            embeddings_array = np.concatenate((embeddings_array, embeddings), axis=0)
            
            embeddings = torch.empty(0, d, device=device)

    embedding_time_end = time.time()
    embeddings = embeddings.float().cpu().detach().numpy()
    faiss.normalize_L2(embeddings)
    embeddings_array = np.concatenate((embeddings_array, embeddings), axis=0)

    return embeddings_array

def compute_all_embeddings_parallel(input_fasta, output_folder, size_of_chunk=1000000, num_gpus=1, resume=False):
    """
    Compute embeddings for sequences in a FASTA file in parallel using multiple GPUs.
    This function reads sequences from a FASTA file, splits them into chunks, and computes
    embeddings in parallel using multiple GPUs. The embeddings and sequence positions are
    saved to binary files in the specified output folder.
    Args:
        input_fasta (str): Path to the input FASTA file containing sequences.
        output_folder (str): Path to the folder where output files will be saved.
        size_of_chunk (int, optional): Number of sequences per chunk for each GPU. Defaults to 1,000,000.
        num_gpus (int, optional): Number of GPUs to use for parallel processing. Defaults to 1.
    Returns:
        None
    Output:
        - A binary file containing embeddings for the sequences, saved as `<input_fasta>.embeddings`.
        - A binary file containing sequence positions, saved as `<input_fasta>.names`.
    Notes:
        - The function uses multiprocessing to distribute the workload across GPUs.
        - Embeddings are computed using the `embed_glm2_parallel` function, which must be defined elsewhere.
        - The embeddings are saved in `float16` format, and sequence positions are saved in `uint64` format.
        - Remaining sequences that do not fit into full chunks are processed at the end.
    """

    chunk_positions_file = os.path.join(output_folder, "chunk_start_positions.txt")
    chunk_already_done_file = os.path.join(output_folder, "chunk_already_done.txt") #chunk already done is a file containing as many bytes as there are chunks, set to 0 or 1
    chunk_start_positions = []
    if not os.path.exists(chunk_positions_file):

        with open(input_fasta) as fi:
            line = fi.readline()
            nb_prot = 0
            while line:
                if ">" == line[0]:
                    if nb_prot % size_of_chunk == 0:
                        chunk_start_positions.append(fi.tell() - len(line))  # Record the start position of the chunk
                    nb_prot += 1

                line = fi.readline()

        # Save chunk start positions to a file
        try:
            with open(chunk_positions_file, "x") as cpf:  # Use "x" mode to ensure the file is created exclusively
                for position in chunk_start_positions:
                    cpf.write(f"{position}\n")
            
        except FileExistsError:
            ...
        print(f"Chunk start positions saved to {chunk_positions_file}")
    else:
        print(f"Chunk start positions file already exists at {chunk_positions_file}")
        if os.path.exists(chunk_positions_file):
            with open(chunk_positions_file, "r") as cpf:
                chunk_start_positions = [int(line.strip()) for line in cpf.readlines()]

    # Ensure the "chunk_already_done.txt" file exists
    if not os.path.exists(chunk_already_done_file):
        with open(chunk_already_done_file, "wb") as cadf:
            fcntl.flock(cadf, fcntl.LOCK_EX)  # Lock the file exclusively
            cadf.write(b'\x00' * len(chunk_start_positions))
            fcntl.flock(cadf, fcntl.LOCK_UN)  # Unlock the file
    else:
        # Wait until we are sure the file is properly created
        with open(chunk_already_done_file, "rb+") as cadf:
            fcntl.flock(cadf, fcntl.LOCK_EX)  # Try to lock the file
            fcntl.flock(cadf, fcntl.LOCK_UN)  # Unlock the file after ensuring it's created

    there_are_still_chunks_to_process = True

    while there_are_still_chunks_to_process:

        chunks_to_process = []

        #decide which chunk to use
        with open(chunk_already_done_file, "rb+") as cadf:
            fcntl.flock(cadf, fcntl.LOCK_EX)  # lock the file

            all_1 = True
            chunk_status = bytearray(cadf.read())
            chunk_status_as_ints = [int(byte) for byte in chunk_status]

            # Take the first 10*num_gpus non-taken-care-of chunks and set them to 2
            chunks_to_process = []
            for i, status in enumerate(chunk_status_as_ints):
                if status == 0:  # Not done
                    chunks_to_process.append(i)
                    if len(chunks_to_process) == 10 * num_gpus:
                        break
                if status != 1:
                    all_1 = False

            # Mark these chunks as "in process" (2)
            for chunk_id in chunks_to_process:
                chunk_status[chunk_id] = 2

            cadf.seek(0)
            cadf.write(bytearray(chunk_status))
            cadf.flush()

            fcntl.flock(cadf, fcntl.LOCK_UN)  # Unlock the file

            if all_1 :
                there_are_still_chunks_to_process = False

            if len(chunks_to_process) == 0 and there_are_still_chunks_to_process:

                cadf.seek(0)
                chunk_status = bytearray(cadf.read())
                chunk_status_as_ints = [int(byte) for byte in chunk_status]

                # Take the first 10*num_gpus 2 chunks and set them to 2
                chunks_to_process = []
                for i, status in enumerate(chunk_status_as_ints):
                    if status == 2:  # In progress
                        chunks_to_process.append(i)
                        if len(chunks_to_process) == 10 * num_gpus:
                            break


        if not there_are_still_chunks_to_process:
            break

        sequences = []
        positions_in_file = []

        # Process the selected chunks
        with open(input_fasta) as fi:

            tasks = []
            for chunk_id in chunks_to_process:
                start_pos = chunk_start_positions[chunk_id]
                end_pos = chunk_start_positions[chunk_id + 1] if chunk_id + 1 < len(chunk_start_positions) else None
                fi.seek(start_pos)
                sequences = []
                positions_in_file = []
                while end_pos is None or fi.tell() < end_pos:
                    line = fi.readline()
                    if not line:
                        break
                    if ">" == line[0]:
                        positions_in_file.append(fi.tell() - len(line))
                    else:
                        sequences.append(line.strip())
                tasks.append((sequences, positions_in_file, chunk_id))

            # Embed sequences in parallel
            time_start_embedding = time.time()
            with Pool(processes=num_gpus) as pool:
                results = pool.map(embed_glm2_parallel, [(task[0], task[2]%num_gpus) for task in tasks])

            # Write embeddings and update chunk status
            for i, (embeddings, task) in enumerate(zip(results, tasks)):
                chunk_id = task[2]

                # Write embeddings and names to separate files for each chunk
                chunk_embedding_file = os.path.join(output_folder, f"chunk_{chunk_id*size_of_chunk}.embeddings")
                chunk_name_file = os.path.join(output_folder, f"chunk_{chunk_id*size_of_chunk}.names")

                if os.path.exists(chunk_embedding_file) and os.path.exists(chunk_name_file): #bizarre, mark the chunk as not completely done
                    with open(chunk_already_done_file, "rb+") as cadf:
                        fcntl.flock(cadf, fcntl.LOCK_EX)  # Lock the file
                        cadf.seek(chunk_id)
                        cadf.write(b'\x02')  # Update the status to 2
                        cadf.flush()
                        fcntl.flock(cadf, fcntl.LOCK_UN)  # Unlock the file

                with open(chunk_embedding_file, "wb") as foe, open(chunk_name_file, "wb") as fon:
                    fcntl.flock(foe, fcntl.LOCK_EX)  # Lock the embeddings file
                    fcntl.flock(fon, fcntl.LOCK_EX)  # Lock the names file
                    for embedding in embeddings:
                        foe.write(np.array(embedding, dtype=np.float16).tobytes())
                    for pos in task[1]:
                        fon.write(np.array(pos, dtype=np.uint64).tobytes())

                    fcntl.flock(foe, fcntl.LOCK_UN)  # Unlock the embeddings file
                    fcntl.flock(fon, fcntl.LOCK_UN)  # Unlock the names file

                # Mark chunk as done (1)
                with open(chunk_already_done_file, "rb+") as cadf:
                    fcntl.flock(cadf, fcntl.LOCK_EX)  # Lock the file
                    cadf.seek(chunk_id)
                    cadf.write(b'\x01')  # Update the status to 1
                    cadf.flush()
                    fcntl.flock(cadf, fcntl.LOCK_UN)  # Unlock the file

            time_end_embedding = time.time()
            print(f"Processed {len(chunks_to_process) * size_of_chunk} sequences in {time_end_embedding - time_start_embedding:.2f} seconds, or {(len(chunks_to_process) * size_of_chunk)/(time_end_embedding - time_start_embedding):.2f} sequences per seconds")

    # Concatenate resulting files in the order of the chunks
    output_file_embeddings = os.path.join(output_folder, f"{os.path.basename(input_fasta)}.embeddings")
    output_file_names = os.path.join(output_folder, f"{os.path.basename(input_fasta)}.names")
    # Lock the two output files before concatenating
    # Check if we should skip concatenation in resume mode
    skip_concatenation = False
    if resume and os.path.exists(output_file_embeddings) and os.path.exists(output_file_names):
        # Calculate expected sizes
        total_vectors = 0
        for chunk_id in range(len(chunk_start_positions)):
            chunk_embedding_file = os.path.join(output_folder, f"chunk_{chunk_id*size_of_chunk}.embeddings")
            if os.path.exists(chunk_embedding_file):
                total_vectors += os.path.getsize(chunk_embedding_file) // (d * 2)
        expected_embeddings_size = total_vectors * d * 2  # float16 = 2 bytes
        expected_names_size = total_vectors * 8  # uint64 = 8 bytes

        actual_embeddings_size = os.path.getsize(output_file_embeddings)
        actual_names_size = os.path.getsize(output_file_names)

        if actual_embeddings_size == expected_embeddings_size and actual_names_size == expected_names_size:
            print("Concatenated files already exist and are the correct size. Skipping concatenation.")
            skip_concatenation = True

    if not skip_concatenation:
        with open(output_file_embeddings, "wb") as foe, open(output_file_names, "wb") as fon:
            fcntl.flock(foe, fcntl.LOCK_EX)  # Lock the embeddings file
            fcntl.flock(fon, fcntl.LOCK_EX)  # Lock the names file
            for chunk_id in range(len(chunk_start_positions)):
                chunk_embedding_file = os.path.join(output_folder, f"chunk_{chunk_id*size_of_chunk}.embeddings")
                chunk_name_file = os.path.join(output_folder, f"chunk_{chunk_id*size_of_chunk}.names")

                with open(chunk_embedding_file, "rb") as cef, open(chunk_name_file, "rb") as cnf:
                    shutil.copyfileobj(cef, foe)
                    shutil.copyfileobj(cnf, fon)


    print(f"Concatenated embeddings saved to {output_file_embeddings}")
    print(f"Concatenated names saved to {output_file_names}")

    return

def process_subdatabase(embedding_file, bytes_per_vector, database_folder, start_index, end_index, subdatabase_id, file_already_done_subdatabase, embedding_dimension):
    """
    Process a subdatabase by reading vectors, training the FAISS index, and saving it.
    """

    # Open the status file once in read/write mode and lock it
    with open(file_already_done_subdatabase, "rb+") as sf:
        fcntl.flock(sf, fcntl.LOCK_EX)
        sf.seek(0)
        status_list = list(sf.read())
        status = status_list[subdatabase_id]
        if status == 1:
            print(f"Subdatabase {subdatabase_id} has already been processed. Skipping.")
            fcntl.flock(sf, fcntl.LOCK_UN)
            return
        if status == 2:
            if 0 in status_list:
                print(f"Subdatabase {subdatabase_id} is in progress but there are unfinished subdatabases. Skipping.")
                fcntl.flock(sf, fcntl.LOCK_UN)
                return
            else:
                print(f"Subdatabase {subdatabase_id} is in progress and no unfinished subdatabases remain. Proceeding.")
        if status == 0:
            # Mark as in progress (2) before starting processing
            sf.seek(subdatabase_id)
            sf.write(b'\x02')
            sf.flush()
        fcntl.flock(sf, fcntl.LOCK_UN)

    local_vectors = np.empty((0, embedding_dimension), dtype=np.float32)
    print("Loading vectors for database ", subdatabase_id, " going from ", start_index, " to ", end_index, flush=True)
    with open(embedding_file, "rb") as ef:
        ef.seek(start_index * bytes_per_vector)
        bytes_data = ef.read((end_index - start_index) * bytes_per_vector)

        # Convert the bytes data into vectors
        num_vectors = len(bytes_data) // bytes_per_vector
        bytes_read = bytes_data[:num_vectors * bytes_per_vector]
        local_vectors = np.frombuffer(bytes_read, dtype=np.float16).reshape(-1, embedding_dimension).astype(np.float32)

    #load the trained index 
    print("loading trained index")
    local_index_db = faiss.read_index(os.path.join(database_folder, "faiss_index_empty.bins"))

    #fill the index
    print("Adding vectors for subdatabase ", subdatabase_id, flush=True)
    local_index_db.add(local_vectors)
    print("Vectors added", flush=True)

    index_file = os.path.join(database_folder, f"faiss_index_{subdatabase_id}.bin")
    tmp_index_file = os.path.join(database_folder, f"faiss_index_{subdatabase_id}.{os.getpid()}.tmp")

    # Save outside shared lock to avoid long lock holds.
    faiss.write_index(local_index_db, tmp_index_file)

    # Finalize atomically under lock.
    with open(file_already_done_subdatabase, "rb+") as sf:
        fcntl.flock(sf, fcntl.LOCK_EX)
        sf.seek(subdatabase_id)
        status = sf.read(1)
        if status == b'\x01':
            fcntl.flock(sf, fcntl.LOCK_UN)
            if os.path.exists(tmp_index_file):
                os.remove(tmp_index_file)
            print(f"Subdatabase {subdatabase_id} has already been processed. Exiting.")
            return

        os.replace(tmp_index_file, index_file)

        # Mark this subdatabase as done (1) in the status file
        sf.seek(subdatabase_id)
        sf.write(b'\x01')
        sf.flush()
        fcntl.flock(sf, fcntl.LOCK_UN)

    print(f"Subdatabase {subdatabase_id} saved to {index_file}")

def create_faiss_database(input_fasta, database_folder, number_of_threads=1, size_of_subdatabases=5000000, resume=False, pca_components=None):
    
    #choice of the index based on https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    total_number_of_vectors = 0
    initial_index_of_subdatabase = 0

    embedding_file = os.path.join(database_folder, f"{os.path.basename(input_fasta)}.embeddings")
    print("Reading from file:", embedding_file)
    
    # Handle PCA if requested
    pca_folder = os.path.join(database_folder, "pca")
    embedding_dimension = d
    if pca_components is not None and pca_components > 0:
        if resume:
            # Check if PCA files already exist
            pca_model_file = os.path.join(pca_folder, "pca_model.pkl")
            pca_stats_file = os.path.join(pca_folder, "pca_stats.json")
            transformed_embedding_file = os.path.join(database_folder, f"{os.path.basename(embedding_file)}.pca")
            
            if os.path.exists(pca_model_file) and os.path.exists(pca_stats_file) and os.path.exists(transformed_embedding_file):
                print(f"PCA files found. Loading existing PCA model with {pca_components} components...")
                pca, _ = load_pca(pca_folder)
                embedding_file = transformed_embedding_file
                embedding_dimension = pca_components
            else:
                print(f"PCA files not found. Computing PCA with {pca_components} components...")
                pca = compute_and_save_pca(embedding_file, database_folder, n_components=pca_components)
                embedding_file = apply_pca_to_embeddings(embedding_file, database_folder, pca, pca_components)
                embedding_dimension = pca_components
        else:
            print(f"Applying PCA with {pca_components} components...")
            pca = compute_and_save_pca(embedding_file, database_folder, n_components=pca_components)
            embedding_file = apply_pca_to_embeddings(embedding_file, database_folder, pca, pca_components)
            embedding_dimension = pca_components
    
    bytes_per_vector = embedding_dimension * 2  # Each float16 is 2 bytes
    vectors = np.empty((0, embedding_dimension), dtype=np.float32)

    # Determine the total number of vectors
    with open(embedding_file, "rb") as ef:
        ef.seek(0, os.SEEK_END)
        total_vectors = ef.tell() // bytes_per_vector

    print("number of vectors: ", total_vectors )

    empty_index_file = os.path.join(database_folder, "faiss_index_empty.bins")
    if not resume or not os.path.exists(empty_index_file):
        # if True : #test the flat index
        #     print("WARNING: indexing using flat index to compare the results")
        #     index = faiss.index_factory(embedding_dimension, "Flat" )
        if total_vectors < 0 :
            print("WARNING: indexing using HNSW because less than 5M vectors")
            index = faiss.index_factory(embedding_dimension, "HNSW,Flat" )
        else:
            # Read the first 100M vectors and train the index
            with open(embedding_file, "rb") as ef:
                training_data_bytes = ef.read(50_000_000 * bytes_per_vector)
                training_data = np.frombuffer(training_data_bytes, dtype=np.float16).reshape(-1, embedding_dimension).astype(np.float32)

            # Initialize the FAISS index
            # index = faiss.index_factory(embedding_dimension, "OPQ64,IVF64k_HNSW,PQ64") #very small index but not so good recall
            # index = faiss.index_factory(embedding_dimension, "IVF262144_HNSW32,SQ8")
            # index = faiss.index_factory(embedding_dimension, "HNSW64,SQ8") #big index, good recall
            # index = faiss.index_factory(512, "SQ8")
            index = faiss.index_factory(embedding_dimension, "Flat" )
            # index = faiss.index_factory(512, "OPQ256,PQ256")


            if not index.is_trained:
                index.train(training_data)

        # Save the empty FAISS index
        faiss.write_index(index, empty_index_file)
        print(f"Empty FAISS index saved to {empty_index_file}")
    
    print("FAISS index trained and saved")

    # Create tasks for each subdatabase
    tasks = []
    # Create a file to record which tasks (subdatabases) have already been done
    subdb_status_file = os.path.join(database_folder, "subdatabase_already_done.txt")
    num_subdbs = (total_vectors + size_of_subdatabases - 1) // size_of_subdatabases
    subdb_done = [False for i in range(num_subdbs)]
    if not os.path.exists(subdb_status_file) or not resume:
        with open(subdb_status_file, "wb") as sf:
            sf.write(b'\x00' * num_subdbs)

    #index
    for subdatabase_id, start_index in enumerate(range(0, total_vectors, size_of_subdatabases)):
        end_index = min(start_index + size_of_subdatabases, total_vectors)
        tasks.append((embedding_file, bytes_per_vector, database_folder, start_index, end_index, subdatabase_id, subdb_status_file, embedding_dimension))

    # Shuffle tasks for random processing order
    random.shuffle(tasks)

    # Process subdatabases in parallel
    with Pool(processes=number_of_threads) as pool:
        pool.starmap(process_subdatabase, tasks)

def process_subdatabase_usearch(embedding_file, bytes_per_vector, database_folder, start_index, end_index, subdatabase_id, file_already_done_subdatabase, embedding_dimension):
    """
    Process a subdatabase by reading vectors, training the USEARCH index, and saving it.
    """

    # Open the status file once in read/write mode and lock it
    with open(file_already_done_subdatabase, "rb+") as sf:
        fcntl.flock(sf, fcntl.LOCK_EX)
        sf.seek(0)
        status_list = list(sf.read())
        status = status_list[subdatabase_id]
        if status == 1:
            print(f"Subdatabase {subdatabase_id} has already been processed. Skipping.")
            fcntl.flock(sf, fcntl.LOCK_UN)
            return
        else:
            # Mark as in progress (2) before starting processing
            sf.seek(subdatabase_id)
            sf.write(b'\x02')
            sf.flush()
        fcntl.flock(sf, fcntl.LOCK_UN)

    local_vectors = np.empty((0, embedding_dimension), dtype=np.float32)
    print("Loading vectors for database ", subdatabase_id, " going from ", start_index, " to ", end_index, flush=True)
    with open(embedding_file, "rb") as ef:
        ef.seek(start_index * bytes_per_vector)
        bytes_data = ef.read((end_index - start_index) * bytes_per_vector)

        # Convert the bytes data into vectors
        num_vectors = len(bytes_data) // bytes_per_vector
        bytes_read = bytes_data[:num_vectors * bytes_per_vector]
        local_vectors = np.frombuffer(bytes_read, dtype=np.float16).reshape(-1, embedding_dimension).astype(np.float32)

    #fill the index
    index_db = Index(
        ndim=embedding_dimension,
        metric='cos',
        expansion_add = 128,
        expansion_search = 128,
        dtype="i8"
        )
    keys = np.arange(0, end_index-start_index)
    index_db.add(vectors=local_vectors, keys=keys)
    print("Vectors added for subdatabase ", subdatabase_id, flush=True)

    index_file = os.path.join(database_folder, f"usearch_index_{subdatabase_id}.bin")
    tmp_index_file = os.path.join(database_folder, f"usearch_index_{subdatabase_id}.{os.getpid()}.tmp")

    # Save the USEARCH index outside the shared status lock to avoid blocking all workers.
    index_db.save(tmp_index_file)

    # Finalize atomically under lock: either keep existing file if already done, or publish this one.
    with open(file_already_done_subdatabase, "rb+") as sf:
        fcntl.flock(sf, fcntl.LOCK_EX)
        sf.seek(subdatabase_id)
        status = sf.read(1)
        if status == b'\x01':
            fcntl.flock(sf, fcntl.LOCK_UN)
            if os.path.exists(tmp_index_file):
                os.remove(tmp_index_file)
            print(f"Subdatabase {subdatabase_id} has already been processed. Exiting.")
            return

        os.replace(tmp_index_file, index_file)

        # Mark this subdatabase as done (1) in the status file
        sf.seek(subdatabase_id)
        sf.write(b'\x01')
        sf.flush()
        fcntl.flock(sf, fcntl.LOCK_UN)
    print(f"Subdatabase {subdatabase_id} saved to {index_file}")

def create_usearch_database(input_fasta, database_folder, number_of_threads=1, size_of_subdatabases=5000000, resume=False, pca_components=None):

    # Similar to create_faiss_database but using USEARCH instead of FAISS
    
    total_number_of_vectors = 0
    initial_index_of_subdatabase = 0

    embedding_file = os.path.join(database_folder, f"{os.path.basename(input_fasta)}.embeddings")
    print("Reading from file for usearch db creation:", embedding_file)
    
    # Handle PCA if requested
    embedding_dimension = d
    if pca_components is not None and pca_components > 0:
        if resume:
            # Check if PCA files already exist
            pca_folder = os.path.join(database_folder, "pca")
            pca_model_file = os.path.join(pca_folder, "pca_model.pkl")
            pca_stats_file = os.path.join(pca_folder, "pca_stats.json")
            transformed_embedding_file = os.path.join(database_folder, f"{os.path.basename(embedding_file)}.pca")
            
            if os.path.exists(pca_model_file) and os.path.exists(pca_stats_file) and os.path.exists(transformed_embedding_file):
                print(f"PCA files found. Loading existing PCA model with {pca_components} components...")
                pca, _ = load_pca(pca_folder)
                embedding_file = transformed_embedding_file
                embedding_dimension = pca_components
            else:
                print(f"PCA files not found. Computing PCA with {pca_components} components for USEARCH...")
                pca = compute_and_save_pca(embedding_file, database_folder, n_components=pca_components)
                embedding_file = apply_pca_to_embeddings(embedding_file, database_folder, pca, pca_components)
                embedding_dimension = pca_components
        else:
            print(f"Applying PCA with {pca_components} components for USEARCH...")
            pca = compute_and_save_pca(embedding_file, database_folder, n_components=pca_components)
            embedding_file = apply_pca_to_embeddings(embedding_file, database_folder, pca, pca_components)
            embedding_dimension = pca_components
    
    bytes_per_vector = embedding_dimension * 2  # Each float16 is 2 bytes
    vectors = np.empty((0, embedding_dimension), dtype=np.float32)
    # Determine the total number of vectors
    with open(embedding_file, "rb") as ef:
        ef.seek(0, os.SEEK_END)
        total_vectors = ef.tell() // bytes_per_vector
    # print("number of vectors: ", total_vectors )

    # Create tasks for each subdatabase
    tasks = []
    # Create a file to record which tasks (subdatabases) have already been done
    subdb_status_file = os.path.join(database_folder, "subdatabase_usearch_already_done.txt")
    num_subdbs = (total_vectors + size_of_subdatabases - 1) // size_of_subdatabases
    subdb_done = [False for i in range(num_subdbs)]
    if not os.path.exists(subdb_status_file) or not resume:
        with open(subdb_status_file, "wb") as sf:
            sf.write(b'\x00' * num_subdbs)

    #index
    for subdatabase_id, start_index in enumerate(range(0, total_vectors, size_of_subdatabases)):
        end_index = min(start_index + size_of_subdatabases, total_vectors)
        tasks.append((embedding_file, bytes_per_vector, database_folder, start_index, end_index, subdatabase_id, subdb_status_file, embedding_dimension))

    # Shuffle tasks for random processing orders
    random.shuffle(tasks)

    # Process subdatabases in parallel
    with Pool(processes=number_of_threads) as pool:
        pool.starmap(process_subdatabase_usearch, tasks)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Tool to embed sequences with gLM2 and/or build a FAISS database.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Embed subcommand
    embed_parser = subparsers.add_parser('embed', help='Compute embeddings for sequences in a FASTA file.')
    embed_parser.add_argument("input_fasta", type=str, help="Path to the input FASTA file.")
    embed_parser.add_argument("output_folder", type=str, help="Path to the output folder where embeddings will be saved.")
    embed_parser.add_argument("--chunk_size", type=int, default=1000000, help="Number of sequences per chunk (default: 1000000).")
    embed_parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs to use (default: all available).")
    embed_parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs to use for multiprocessing (default: 1).")
    embed_parser.add_argument("-F", "--force", action="store_true", help="Force overwrite of the output folder if it exists.")
    embed_parser.add_argument("--resume", action="store_true", help="Resume the embedding process if interrupted.")

    # FAISS subcommand
    faiss_parser = subparsers.add_parser('faiss', help='Create FAISS database from embeddings.')
    faiss_parser.add_argument("input_fasta", type=str, help="Path to the input FASTA file (used for naming).")
    faiss_parser.add_argument("database_folder", type=str, help="Path to the folder containing embeddings and where FAISS DB will be saved.")
    faiss_parser.add_argument("--subdatabases_size", type=int, default=10000000, help="Number of vectors in each faiss subdatabase (default: 10000000).")
    faiss_parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPU threads to use for building subdatabases (default: 1).")
    faiss_parser.add_argument("--pca_components", type=int, default=None, help="Number of PCA components to reduce dimensions to (default: None, no PCA). If specified, embeddings will be reduced to this many dimensions using PCA.")
    faiss_parser.add_argument("-F", "--force", action="store_true", help="Force overwrite of database files if desired (not applied automatically).")
    faiss_parser.add_argument("--resume", action="store_true", help="Resume the FAISS creation process if interrupted.")

    #USearch subcommand
    usearch_parser = subparsers.add_parser('usearch', help='Create USEARCH database from embeddings.')
    usearch_parser.add_argument("input_fasta", type=str, help="Path to the input FASTA file (used for naming).")
    usearch_parser.add_argument("database_folder", type=str, help="Path to the folder containing embeddings and where USEARCH DB will be saved.")
    usearch_parser.add_argument("--subdatabases_size", type=int, default=10_000_000, help="Number of vectors in each usearch subdatabase (default: 10000000).")
    usearch_parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPU threads to use for building subdatabases (default: 1).")
    usearch_parser.add_argument("--pca_components", type=int, default=None, help="Number of PCA components to reduce dimensions to (default: None, no PCA). If specified, embeddings will be reduced to this many dimensions using PCA.")
    usearch_parser.add_argument("-F", "--force", action="store_true", help="Force overwrite of database files if desired (not applied automatically).")
    usearch_parser.add_argument("--resume", action="store_true", help="Resume the USEARCH creation process if interrupted.")

    args = parser.parse_args()

    # Print GPU info (useful for embed)
    try:
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
    except Exception:
        print("Could not determine GPU count")

    # Set multiprocessing start method based on command
    # - 'spawn' for GPU commands (embed): safer CUDA initialization
    # - 'fork' for CPU commands (faiss, usearch): more efficient, no GPU context issues
    try:
        if args.command == "embed":
            multiprocessing.set_start_method('spawn', force=True)
        else:  # faiss or usearch
            if sys.platform != 'win32':
                multiprocessing.set_start_method('fork', force=True)
    except RuntimeError as e:
        # already set
        pass

    print(f"Version: {__version__}")
    print(f"Command: {args.command}")

    if args.command == "embed":
        # determine GPUs count if 0 => use all
        if args.num_gpus == 0:
            args.num_gpus = torch.cuda.device_count()

        # handle output folder existence
        if os.path.exists(args.output_folder):
            if not args.force and not args.resume:
                print(f"Output folder '{args.output_folder}' already exists. Use --force to overwrite.")
                sys.exit(1)
            elif not args.resume:
                print("Warning: overwriting a previously existing database")
                shutil.rmtree(args.output_folder)

        if not args.resume or not os.path.exists(args.output_folder):
            os.makedirs(args.output_folder, exist_ok=True)

        start_time_embeddings = time.time()
        compute_all_embeddings_parallel(
            input_fasta=args.input_fasta,
            output_folder=args.output_folder,
            size_of_chunk=args.chunk_size,
            num_gpus=args.num_gpus,
            resume=args.resume
        )
        end_time_embeddings = time.time()
        print(f"Time taken to compute all embeddings: {end_time_embeddings - start_time_embeddings:.2f} seconds")
        print("Embedding step done!")

    elif args.command == "faiss":
        # call FAISS creation
        # database_folder is the folder containing embeddings (output of embed) and where FAISS files will be written
        start_time_faiss = time.time()
        create_faiss_database(
            input_fasta=args.input_fasta,
            database_folder=args.database_folder,
            number_of_threads=args.num_cpus,
            size_of_subdatabases=args.subdatabases_size,
            resume=args.resume,
            pca_components=args.pca_components
        )
        end_time_faiss = time.time()
        print(f"Time taken to create FAISS database: {end_time_faiss - start_time_faiss:.2f} seconds")
        print("FAISS database creation done!")

    elif args.command == "usearch":
        # call USEARCH creation
        # database_folder is the folder containing embeddings (output of embed) and where USEARCH files will be written
        start_time_usearch = time.time()
        create_usearch_database(
            input_fasta=args.input_fasta,
            database_folder=args.database_folder,
            number_of_threads=args.num_cpus,
            size_of_subdatabases=args.subdatabases_size,
            resume=args.resume,
            pca_components=args.pca_components
        )
        end_time_usearch = time.time()
        print(f"Time taken to create USEARCH database: {end_time_usearch - start_time_usearch:.2f} seconds")
        print("USEARCH database creation done!")
