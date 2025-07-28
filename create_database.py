import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, T5Tokenizer, BertConfig
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.lines import Line2D
import random
import argparse
import os
import fcntl
import faiss
import time
import multiprocessing
from multiprocessing import Pool, Manager
from functools import partial
import shutil

d = 512

__version__ = "1.0.0"

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
    
    model = AutoModel.from_pretrained('tattabio/gLM2_650M_embed', torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M_embed', trust_remote_code=True)
    
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

    device = torch.device(f'cuda:{device_id}' if device_id != "cpu" else "cpu")
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

def process_subdatabase(embedding_file, bytes_per_vector, database_folder, start_index, end_index, subdatabase_id, file_already_done_subdatabase):
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

    local_vectors = np.empty((0, d), dtype=np.float32)
    print("Loading vectors for database ", subdatabase_id, " going from ", start_index, " to ", end_index, flush=True)
    with open(embedding_file, "rb") as ef:
        ef.seek(start_index * bytes_per_vector)
        bytes_data = ef.read((end_index - start_index) * bytes_per_vector)

        # Convert the bytes data into vectors
        num_vectors = len(bytes_data) // bytes_per_vector
        bytes_read = bytes_data[:num_vectors * bytes_per_vector]
        local_vectors = np.frombuffer(bytes_read, dtype=np.float16).reshape(-1, d).astype(np.float32)

    #load the trained index 
    print("loading trained index")
    local_index_db = faiss.read_index(os.path.join(database_folder, "faiss_index_empty.bins"))

    #fill the index
    print("Adding vectors for subdatabase ", subdatabase_id, flush=True)
    local_index_db.add(local_vectors)
    print("Vectors added", flush=True)

    # Lock the status file, checkflush=True status, and write the FAISS index atomically
    with open(file_already_done_subdatabase, "rb+") as sf:
        fcntl.flock(sf, fcntl.LOCK_EX)
        sf.seek(subdatabase_id)
        status = sf.read(1)
        if status == b'\x01':
            print(f"Subdatabase {subdatabase_id} has already been processed. Exiting.")
            fcntl.flock(sf, fcntl.LOCK_UN)
            return

        # Save the FAISS index for this subdatabase while holding the lock
        index_file = os.path.join(database_folder, f"faiss_index_{subdatabase_id}.bin")
        faiss.write_index(local_index_db, index_file)

        # Mark this subdatabase as done (1) in the status file
        sf.seek(subdatabase_id)
        sf.write(b'\x01')
        sf.flush()
        fcntl.flock(sf, fcntl.LOCK_UN)

    print(f"Subdatabase {subdatabase_id} saved to {index_file}")

def create_faiss_database(input_fasta, database_folder, number_of_threads=1, size_of_subdatabases=5000000, resume=False):
    
    #choice of the index based on https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    total_number_of_vectors = 0
    initial_index_of_subdatabase = 0

    embedding_file = os.path.join(database_folder, f"{os.path.basename(input_fasta)}.embeddings")
    print("Reading from file:", embedding_file)
    bytes_per_vector = d * 2  # Each float16 is 2 bytes
    vectors = np.empty((0, d), dtype=np.float32)

    # Determine the total number of vectors
    with open(embedding_file, "rb") as ef:
        ef.seek(0, os.SEEK_END)
        total_vectors = ef.tell() // bytes_per_vector

    print("number of vectors: ", total_vectors )

    empty_index_file = os.path.join(database_folder, "faiss_index_empty.bins")
    if not resume or not os.path.exists(empty_index_file):
        if total_vectors < 50000000 :
            print("WARNING: indexing using HNSW because less than 5M vectors")
            index = faiss.index_factory(d, "HNSW,Flat" )
        else:
            # Read the first 5M vectors and train the index
            with open(embedding_file, "rb") as ef:
                training_data_bytes = ef.read(5000000 * bytes_per_vector)
                training_data = np.frombuffer(training_data_bytes, dtype=np.float16).reshape(-1, d).astype(np.float32)

            # Initialize the FAISS index
            index = faiss.index_factory(d, "OPQ64,IVF64k_HNSW,PQ64")
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
        tasks.append((embedding_file, bytes_per_vector, database_folder, start_index, end_index, subdatabase_id, subdb_status_file))

    # Process subdatabases in parallel
    with Pool(processes=number_of_threads) as pool:
        pool.starmap(process_subdatabase, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embeddings for sequences in a FASTA file using gLM2 model.")
    parser.add_argument("input_fasta", type=str, help="Path to the input FASTA file.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder where embeddings will be saved.")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Number of sequences per chunk (default: 100000).")
    parser.add_argument("--subdatabases_size", type=int, default=10000000, help="Number of vectors in each faiss database")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs to use (default: all available).")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of GPUs to use (default: 1).")
    parser.add_argument("-F", "--force", action="store_true", help="Force overwrite of the output folder if it exists.")
    parser.add_argument("--resume", action="store_true", help="Resume the embedding process if interrupted.")
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')

    args = parser.parse_args()
    print(f"Number of available GPUs: {torch.cuda.device_count()}")

    if args.num_gpus == 0 :
        args.num_gpus = torch.cuda.device_count()

    multiprocessing.set_start_method('spawn') #to work with CUDA

    # if os.path.exists(args.output_folder):
    #     if not args.force and not args.resume:
    #         print(f"Output folder '{args.output_folder}' already exists. Use --force to overwrite.")
    #         sys.exit(1)
    #     elif not args.resume:
    #         print("Warning: overwriting a previously existing database")
    #         shutil.rmtree(args.output_folder)  

    if not args.resume or not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    # start_time_embeddings = time.time()
    # compute_all_embeddings_parallel(
    #     input_fasta=args.input_fasta,
    #     output_folder=args.output_folder,
    #     size_of_chunk=args.chunk_size,
    #     num_gpus=args.num_gpus,
    #     resume=args.resume
    # )
    # end_time_embeddings = time.time()
    # print(f"Time taken to compute all embeddings: {end_time_embeddings - start_time_embeddings:.2f} seconds")

    start_time_faiss = time.time()
    create_faiss_database(args.input_fasta, database_folder=args.output_folder, number_of_threads=args.num_cpus, size_of_subdatabases=args.subdatabases_size, resume=args.resume)
    end_time_faiss = time.time()
    print(f"Time taken to create FAISS database: {end_time_faiss - start_time_faiss:.2f} seconds")

    print("All done!")



