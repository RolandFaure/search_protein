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
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    
    model = AutoModel.from_pretrained('tattabio/gLM2_650M_embed', torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained('tattabio/gLM2_650M_embed', trust_remote_code=True)
    
    return model, tokenizer

def embed_glm2_parallel(sequences_device_id):
    """
    Generates embeddings for a list of sequences in parallel using a specified device.
    Args:
        sequences_device_id (tuple): A tuple containing:
            - sequences (list of str): A list of input sequences to embed.
            - device_id (int): The ID of the GPU device to use for embedding.
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
        sequences[i] = "<+>" + sequences[i]

    device = torch.device(f'cuda:{device_id}')
    embeddings_array = np.empty((0, d), dtype=np.float32)
    embeddings = torch.empty(0, d, device=device)
    batch_size = 50

    end_time = time.time()
    embedding_time_start = time.time()

    for seq_start in range(0, len(sequences), batch_size):
        encodings = tokenizer(sequences[seq_start:min(seq_start + batch_size, len(sequences))], return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            attention_mask = encodings.attention_mask.bool().cuda() #this is very important to handle the padding correctly
            pooled_embeds = model(encodings.input_ids.to(device_id), attention_mask=attention_mask).pooler_output

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

def compute_all_embeddings_parallel(input_fasta, output_folder, size_of_chunk=1000000, num_gpus=1):
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
    
    name = ""
    index = 0
    sequences = []
    position_in_file = []
    manager = Manager()
    gpu_queue = manager.Queue()

    for gpu_id in range(num_gpus):
        gpu_queue.put(gpu_id)

    output_file_embeddings = os.path.join(output_folder, f"{os.path.basename(input_fasta)}.embeddings")
    output_file_names = os.path.join(output_folder, f"{os.path.basename(input_fasta)}.names")

    with open(output_file_embeddings, "ab") as foe , open(output_file_names, "ab") as fon, open(input_fasta) as fi:

        line = fi.readline()
        while line:
            if ">" == line[0]:
                position_in_file.append(fi.tell()-len(line))
            else:
                sequences.append(line.strip())
                index += 1

                if index % (size_of_chunk*num_gpus) == 0:

                    chunk_sequences = [sequences[i*size_of_chunk : (i+1)*size_of_chunk] for i in range(num_gpus)]

                    time_start_embedding = time.time()
                    with Pool(processes=num_gpus) as pool:
                        embeddings_chunks = pool.map(embed_glm2_parallel, [(chunk_sequences[i], i) for i in range(num_gpus)])
                        
                    embeddings = np.empty((0, d), dtype=np.float32)
                    for chunk in embeddings_chunks:
                        embeddings = np.concatenate((embeddings, chunk), axis=0)

                        for embedding in embeddings:
                            foe.write(np.array(embedding, dtype=np.float16).tobytes())
                        for pos in position_in_file:
                            fon.write(np.array(pos, dtype=np.uint64).tobytes())

                    time_end_embedding=time.time()
                    print("Embedded " + str(size_of_chunk*num_gpus) + " sequences in " + str(time_end_embedding-time_start_embedding) + " seconds, which makes " + str(size_of_chunk*num_gpus/(time_end_embedding-time_start_embedding)) + " prot/sec")

                    sequences = []
                    position_in_file = []

            line = fi.readline()

        #finish with the remaining sequences
        if len(sequences) > 0:
            device_id = gpu_queue.get()
            embeddings = embed_glm2_parallel((sequences, device_id))
            gpu_queue.put(device_id)

            for embedding in embeddings:
                foe.write(np.array(embedding, dtype=np.float16).tobytes())
            for pos in position_in_file:
                fon.write(np.array(pos, dtype=np.uint64).tobytes())

    return

def create_faiss_database(input_fasta, database_folder, size_of_subdatabases=5000000):
    
    #choice of the index based on https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    index_db = faiss.index_factory(d, "OPQ64,IVF64k_HNSW,PQ64" )
    total_number_of_vectors = 0
    initial_index_of_subdatabase = 0

    size_of_training_data = 200000
    if size_of_training_data > size_of_subdatabases:
        print("ERROR, code 4509")
        sys.exit(1)

    # print("DEBUG INDEX")
    # index_db = faiss.index_factory(d, "HNSW,Flat" )

    embedding_file = os.path.join(database_folder, f"{os.path.basename(input_fasta)}.embeddings")
    print("Reading from file:", embedding_file)
    with open(embedding_file, "rb") as ef:
        bytes_data = ef.read()

    # Cut the bytes data into pieces of two bytes and reshape into vectors of 512 floats
    bytes_per_vector = d * 2  # Each float16 is 2 bytes
    num_vectors = len(bytes_data) // bytes_per_vector
    bytes_read = bytes_data[:num_vectors * bytes_per_vector]
    vector = np.frombuffer(bytes_read, dtype=np.float16).reshape(-1, d).astype(np.float32)
    vectors = np.empty((0, d), dtype=np.float32)
    vectors = np.vstack((vectors, vector))

    if len(vectors) >= size_of_training_data:
        if not index_db.is_trained:
            index_db.train(vectors)
        total_number_of_vectors += len(vectors)
        index_db.add(vectors)
        vectors = np.empty((0, d), dtype=np.float32)

        if total_number_of_vectors-initial_index_of_subdatabase >= size_of_subdatabases:

            # Save the current FAISS index to a file
            index_file = os.path.join(database_folder, f"faiss_index_{initial_index_of_subdatabase}.bin")
            faiss.write_index(index_db, index_file)
            print(f"Subdatabase {initial_index_of_subdatabase} saved to {index_file}")

            # Start a new subdatabase, keeping the same index structure
            initial_index_of_subdatabase = total_number_of_vectors
            index_db.reset()
            vectors = np.empty((0, d), dtype=np.float32)

    # Add any remaining vectors
    if len(vectors) > 0:
        if not index_db.is_trained:
            index_db.train(vectors)
        index_db.add(vectors)

    # Save the FAISS index
    index_file = os.path.join(database_folder, f"faiss_index_{initial_index_of_subdatabase}.bin")
    faiss.write_index(index_db, index_file)
    print(f"FAISS database created and saved to {index_file}")

def query_faiss_database(input_fasta, database_folder, query_sequence, cutoff=0.2, num_gpus=1):
    """
    Queries a FAISS database with a set of sequences and retrieves the nearest neighbors.

    Args:
        database_folder (str): Path to the folder containing FAISS database files (.bin).
        query_sequences (list of str): List of sequences to query.
        num_gpus (int, optional): Number of GPUs to use for embedding the query sequences. Defaults to 1.

    Returns:
        list of list of tuple: A list where each element corresponds to a query sequence and contains
                                a list of tuples (index, distance) for the nearest neighbors.
    """

    # Embed the query sequences
    print("Embedding query sequences...")
    query_embeddings = embed_glm2_parallel(([query_sequence], 0))

    # Load all FAISS indices from the database folder
    print("Querying...")
    results = []
    faiss_indices = []
    for file in os.listdir(database_folder):
        if file.endswith(".bin"):

            file_starting_pos = int(file.strip(".bin").split("_")[2])

            index_path = os.path.join(database_folder, file)
            index = faiss.read_index(index_path)
            nb_of_searches = 1
            distances = []
            while len(distances) == 0 or distances[0][-1] < cutoff or nb_of_searches > 10 : #loop to increase the number of matches if everything matches
                distances, indices = index.search(query_embeddings, k=20*2**(nb_of_searches))
                nb_of_searches+=1
            results.append([(int(indices[0][j]+file_starting_pos), float(distances[0][j])) for j in range(len(indices[0])) if distances[0][j] < cutoff])

            # Merge and sort all results by distance
            merged_results = []
            for result in results:
                merged_results.extend(result)
            merged_results.sort(key=lambda x: x[1])  # Sort by distance (second element of tuple)

            # Open the .embeddings file and associate names to distances
            final_results = []
            names_file = os.path.join(database_folder, f"{os.path.basename(input_fasta)}.names")

            with open(names_file, "rb") as nf, open(input_fasta, "r") as fastafile:
                
                for result in merged_results :
                    name_positions = []
                    nf.seek(8*result[0])
                    position_name = int.from_bytes(nf.read(8), byteorder='little', signed=False)

                    fastafile.seek(position_name)
                    name_line = fastafile.readline().strip()
                    final_results.append((name_line, result[1]))

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embeddings for sequences in a FASTA file using gLM2 model.")
    parser.add_argument("input_fasta", type=str, help="Path to the input FASTA file.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder where embeddings will be saved.")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Number of sequences per chunk (default: 100000).")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs to use (default: all available).")
    parser.add_argument("-F", "--force", action="store_true", help="Force overwrite of the output folder if it exists.")

    args = parser.parse_args()
    print(f"Number of available GPUs: {torch.cuda.device_count()}")

    if args.num_gpus == 0 :
        args.num_gpus = torch.cuda.device_count()

    multiprocessing.set_start_method('spawn') #to work with CUDA

    if os.path.exists(args.output_folder):
        if not args.force:
            print(f"Output folder '{args.output_folder}' already exists. Use --force to overwrite.")
            sys.exit(1)
        else:
            print("Warning: overwriting a previously existing database")
        shutil.rmtree(args.output_folder)  
    os.makedirs(args.output_folder)

    compute_all_embeddings_parallel(
        input_fasta=args.input_fasta,
        output_folder=args.output_folder,
        size_of_chunk=args.chunk_size,
        num_gpus=args.num_gpus
    )

    create_faiss_database(args.input_fasta, database_folder=args.output_folder, size_of_subdatabases=5000000)

    query_results = query_faiss_database(args.input_fasta, database_folder=args.output_folder, query_sequence = "MPPHAARPGPAQNRRGCAMAVMTPRRERSSLLSRALQVTAAAATALVTAVSLAAPAHAANPYERGPNPTDALLEARSGPFSVSEENVSRLGASGFGGGTIYYPRENNTYGAVAISPGYTGTQASVAWLGKRIASHGFVVITIDTITTLDQPDSRARQLNAALDYMINDASSAVRSRIDSSRLAVMGHSMGGGGSLRLASQRPDLKAAIPLTPWHLNKNWSSVRVPTLIIGADLDTIAPVLTHARPFYNSLPTSISKAYLELDGATHFAPNIPNKIIGKYSVAWLKRFVDNDTRYTQFLCPGPRDGLFGEVEEYRSTCPF")
    print(query_results)

