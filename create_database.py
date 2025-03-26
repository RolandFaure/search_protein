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

def process_subdatabase(embedding_file, bytes_per_vector, database_folder, start_index, end_index, subdatabase_id):
    """
    Process a subdatabase by reading vectors, training the FAISS index, and saving it.
    """

    local_vectors = np.empty((0, d), dtype=np.float32)
    with open(embedding_file, "rb") as ef:
        ef.seek(start_index * bytes_per_vector)
        bytes_data = ef.read((end_index - start_index) * bytes_per_vector)

        # Convert the bytes data into vectors
        num_vectors = len(bytes_data) // bytes_per_vector
        bytes_read = bytes_data[:num_vectors * bytes_per_vector]
        local_vectors = np.frombuffer(bytes_read, dtype=np.float16).reshape(-1, d).astype(np.float32)

    #load the trained index 
    local_index_db = faiss.read_index(os.path.join(database_folder, "faiss_index_empty.bins"))

    #fill the index
    local_index_db.add(local_vectors)

    # Save the FAISS index for this subdatabase
    index_file = os.path.join(database_folder, f"faiss_index_{subdatabase_id}.bin")
    faiss.write_index(local_index_db, index_file)
    print(f"Subdatabase {subdatabase_id} saved to {index_file}")

def create_faiss_database(input_fasta, database_folder, number_of_threads=1, size_of_subdatabases=5000000):
    
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
    empty_index_file = os.path.join(database_folder, "faiss_index_empty.bins")
    faiss.write_index(index, empty_index_file)
    print(f"Empty FAISS index saved to {empty_index_file}")

    # Create tasks for each subdatabase
    tasks = []
    for subdatabase_id, start_index in enumerate(range(0, total_vectors, size_of_subdatabases)):
        end_index = min(start_index + size_of_subdatabases, total_vectors)
        tasks.append((embedding_file, bytes_per_vector, database_folder, start_index, end_index, subdatabase_id))

    # Process subdatabases in parallel
    with Pool(processes=number_of_threads) as pool:
        pool.starmap(process_subdatabase, tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute embeddings for sequences in a FASTA file using gLM2 model.")
    parser.add_argument("input_fasta", type=str, help="Path to the input FASTA file.")
    parser.add_argument("output_folder", type=str, help="Path to the output folder where embeddings will be saved.")
    parser.add_argument("--chunk_size", type=int, default=1000000, help="Number of sequences per chunk (default: 100000).")
    parser.add_argument("--num_gpus", type=int, default=0, help="Number of GPUs to use (default: all available).")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of GPUs to use (default: 1).")
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

    start_time_embeddings = time.time()
    compute_all_embeddings_parallel(
        input_fasta=args.input_fasta,
        output_folder=args.output_folder,
        size_of_chunk=args.chunk_size,
        num_gpus=args.num_gpus
    )
    end_time_embeddings = time.time()
    print(f"Time taken to compute all embeddings: {end_time_embeddings - start_time_embeddings:.2f} seconds")

    start_time_faiss = time.time()
    create_faiss_database(args.input_fasta, database_folder=args.output_folder, number_of_threads=args.num_cpus, size_of_subdatabases=10000000)
    end_time_faiss = time.time()
    print(f"Time taken to create FAISS database: {end_time_faiss - start_time_faiss:.2f} seconds")

    print("All done!")

