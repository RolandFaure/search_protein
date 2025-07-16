from create_database import embed_glm2_parallel
import faiss
import argparse
import os
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import numpy as np
import sys

__version__= "1.0.1"


def query_bin(bin_file, input_fasta, database_folder, query_embeddings, query_names, subdatabase_size, cutoff):
    start_time = time.time()
    faiss.omp_set_num_threads(1)

    file_starting_pos = int(bin_file.strip(".bin").split("_")[2])*subdatabase_size
    index_path = os.path.join(database_folder, bin_file)
    # Load the FAISS index inside the function so each process loads its own copy
    index = faiss.read_index(index_path)
    nb_of_searches = 1
    distances = []
    while len(distances) == 0 or (distances[0][-1] < cutoff and nb_of_searches <= 10):
        k = 20 * 2 ** (nb_of_searches)
        distances, indices = index.search(query_embeddings, k=k)
        nb_of_searches += 1

    results = [[] for _ in range(len(distances))]
    for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for dist, idx in zip(dist_row, idx_row):
            if dist < cutoff:
                results[i].append((file_starting_pos + idx, dist))
                print("in index ", bin_file, " found ", idx, " with a distance of ", dist)

    results_sorted = [sorted(result, key=lambda x: x[1]) for result in results]
    final_results = []
    names_file = os.path.join(database_folder, f"{os.path.basename(input_fasta)}.names")
    with open(names_file, "rb") as nf, open(input_fasta, "r") as fastafile:
        for query_idx in range(len(results_sorted)):
            for result in results_sorted[query_idx]:
                nf.seek(8 * result[0])
                position_name = int.from_bytes(nf.read(8), byteorder='little', signed=False)
                fastafile.seek(position_name)
                name_line = fastafile.readline().strip()
                sequence_line = fastafile.readline().strip()
                final_results.append((query_names[query_idx], name_line, sequence_line, result[1]))

    elapsed_time = time.time() - start_time
    print(f"Time taken for querying bin {bin_file}: {elapsed_time:.2f} seconds")
    return final_results

def parallel_query_bins(input_fasta, database_folder, query_embeddings, query_names, cutoff=0.2, subdatabase_size=10000000, max_workers=4):
    bin_files = [file for file in os.listdir(database_folder) if file.endswith(".bin")]

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Copy query_embeddings for each process to avoid shared memory issues
        futures = [
            executor.submit(
            query_bin, bin_file, input_fasta, database_folder, query_embeddings, query_names, subdatabase_size, cutoff
            )
            for bin_file in bin_files
        ]
        for future in as_completed(futures):
            all_results.extend(future.result())
    return all_results

def query_faiss_database(input_fasta, database_folder, query_sequences, query_names, cutoff=0.2, num_gpus=1, gpus_available=True, subdatabase_size=10000000, max_workers=4):
    """
    Queries a FAISS database with a set of sequences and retrieves the nearest neighbors in parallel.

    Args:
        input_fasta (str): Path to the original FASTA file.
        database_folder (str): Path to the folder containing FAISS database files (.bin).
        query_sequences (list of str): List of sequences to query.
        num_gpus (int, optional): Number of GPUs to use for embedding the query sequences. Defaults to 1.
        gpus_available (bool, optional): Whether GPUs are available. Defaults to True.
        max_workers (int, optional): Number of parallel workers. Defaults to 4.

    Returns:
        list of tuple: Each tuple contains (query_idx, name_line, sequence_line, distance).
    """
    # Embed the query sequences
    start_embedding_time = time.time()
    print("Embedding query sequences...")
    batch_size = 10
    query_embeddings_list = []
    for i in range(0, len(query_sequences), batch_size):
        batch = query_sequences[i:i+batch_size]
        if gpus_available:
            try:
                embeddings = embed_glm2_parallel((batch, 0))
            except RuntimeError as e:
                print(f"GPU embedding failed with error: {e}. Falling back to CPU.")
                embeddings = embed_glm2_parallel((batch, "cpu"))
        else:
            embeddings = embed_glm2_parallel((batch, "cpu"))
        query_embeddings_list.append(embeddings)
    query_embeddings = np.concatenate(query_embeddings_list, axis=0)
    end_embedding_time = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false" #just to be safe and make sure tokenizes is not still parallel

    print("Querying in parallel...")
    # Ensure query_embeddings is a CPU numpy array for process pickling
    if hasattr(query_embeddings, "cpu"):
        query_embeddings = query_embeddings.cpu()
    if hasattr(query_embeddings, "numpy"):
        query_embeddings = query_embeddings.numpy()
    results = parallel_query_bins(
        input_fasta=input_fasta,
        database_folder=database_folder,
        query_embeddings=query_embeddings,
        query_names=query_names,
        cutoff=cutoff,
        subdatabase_size = subdatabase_size,
        max_workers=max_workers
    )

    total_time = time.time() - start_embedding_time
    print(f"Total time taken: {total_time:.2f} seconds")
    return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Query a FAISS database with a sequence.")
    parser.add_argument("--database", required=True, help="Path to the folder containing FAISS database files.")
    parser.add_argument("--input_fasta", required=True, help="Path to the original FASTA file.")
    parser.add_argument("--query_sequences", required=True, help="Fasta file of queris")
    parser.add_argument("--force_cpu", action="store_true", help="Force the use of CPU even if GPUs are available.")
    parser.add_argument("--output", required=False, help="Path to the output FASTA file. If not provided, results will be printed to stdout.")
    parser.add_argument("-t", "--num_threads", type=int, default=1, help="Number of threads to use for parallel querying.")
    parser.add_argument("--subdatabases_size", type=int, default=10000000, help="Number of vectors in each faiss database")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Check if GPUs are available
    if not args.force_cpu:
        try:
            gpus_available = torch.cuda.is_available()
            print("GPU available, moving on")
        except ImportError:
            gpus_available = False
            print("No GPU available, using CPUs...this could be slow if you have many queries")
    else:
        gpus_available = False

    # Read all sequences from the query FASTA file
    query_sequences = []
    query_names = []
    sequence_now = ""
    with open(args.query_sequences, "r") as query_file:
        for line in query_file:
            if line.startswith(">"):
                if sequence_now:
                    query_sequences.append(sequence_now)
                    sequence_now = ""
                query_names.append(line.lstrip('>'))
            else:
                sequence_now += line.strip()
        if sequence_now:
            query_sequences.append(sequence_now)

    query_results = query_faiss_database(
        input_fasta=args.input_fasta,
        database_folder=args.database,
        query_sequences=query_sequences,
        query_names=query_names,
        gpus_available=gpus_available,
        cutoff = 0.2,
        subdatabase_size=args.subdatabases_size,
        max_workers=args.num_threads
    )

    # Sort results by query index and then by ascending distance
    query_results.sort(key=lambda x: (x[0], x[3]))

    if args.output:
        with open(args.output, "w") as output_file:
            output_file.write("#query_name\thit_name\tcosine_distance\thit_sequence\n")
            for query, name, sequence, distance in query_results:
                # Extract hit name (remove '>' if present)
                hit_name = name.strip().lstrip('>').split(' ')[0]
                query_name = query.strip().lstrip('>').split(' ')[0]
                output_file.write(f"{query_name}\t{hit_name}\t{distance}\t{sequence}\n")
    else:
        # Print header
        print("#query_name\thit_name\tcosine_distance\thit_sequence")
        for query, name, sequence, distance in query_results:
            hit_name = name.strip().lstrip('>').split(' ')[0]
            query_name = query.strip().lstrip('>').split(' ')[0]
            print(f"{query_name}\t{hit_name}\t{distance}\t{sequence}")

