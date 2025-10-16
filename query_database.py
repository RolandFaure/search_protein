from create_database import embed_glm2_parallel
import faiss
import argparse
import os
import torch
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import numpy as np
import sys
import tempfile
import subprocess
from collections import defaultdict
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import csv

__version__= "1.1.1"


def query_bin(bin_file, original_fasta, database_folder, query_embeddings, query_names, subdatabase_size, cutoff):
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

    results_sorted = [sorted(result, key=lambda x: x[1]) for result in results]
    final_results = []
    names_file = os.path.join(database_folder, f"{os.path.basename(original_fasta)}.names")
    with open(names_file, "rb") as nf, open(original_fasta, "r") as fastafile:
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

def parallel_query_bins(original_fasta, database_folder, query_embeddings, query_names, cutoff=0.2, subdatabase_size=10000000, max_workers=4):
    faiss_database_folder = database_folder+"/faiss"
    bin_files = [file for file in os.listdir(faiss_database_folder) if file.endswith(".bin")]

    all_results = []
    # for bin_file in bin_files:
    #     res = query_bin(bin_file, original_fasta, faiss_database_folder, query_embeddings, query_names, subdatabase_size, cutoff)
    #     all_results += [res]
    # Ensure query_embeddings is a numpy array for pickling
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(query_bin, bin_file, original_fasta, faiss_database_folder, query_embeddings, query_names, subdatabase_size, cutoff) for bin_file in bin_files
        ]
        for future in as_completed(futures):
            all_results.extend(future.result())
    return all_results

def query_faiss_database(original_fasta, database_folder, query_sequences, query_names, cutoff=0.2, num_gpus=1, gpus_available=True, subdatabase_size=10000000, max_workers=4):
    """
    Queries a FAISS database with a set of sequences and retrieves the nearest neighbors in parallel.

    Args:
        original_fasta (str): Path to the original FASTA file.
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
        original_fasta=original_fasta,
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

def obtain_all_proteins(centroids, database_all_proteins, path_to_centroid_to_prots, num_threads):
    """
    For each result, runs an external command to extract proteins and parses the output.
    Parallelization is limited by the external command and file I/O, but can help if num_threads > 1.
    """

    def process_result(centroid_name):
        centroid_id = centroid_name.strip().lstrip('>').split(' ')[0]
        tmp_filename = f"tmp_{threading.get_ident()}_{centroid_id}.fa"
        command = f"{path_to_centroid_to_prots} {database_all_proteins} {centroid_id} | awk -F'>' '{{ if ($1==\"\") {{ if (NF>=3) print \">\"$2; else print $0 }} else {{ print $1 }} }}' > {tmp_filename}"
        subprocess.run(command, shell=True, check=True)
        proteins = []
        with open(tmp_filename, "r") as tmp_file:
            name = None
            seq = ""
            for line in tmp_file:
                line = line.strip()
                if line.startswith(">"):
                    if name is not None and seq:
                        proteins.append((name, seq))
                    name = line
                    seq = ""
                else:
                    seq = line
            if name is not None and seq:
                proteins.append((name, seq))
        os.remove(tmp_filename)

        return proteins

    all_results = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(process_result, centroid) for centroid in centroids]
        for i, future in enumerate(as_completed(futures), 1):
            proteins = future.result()
            all_results.extend(proteins)
        all_results = list(set(all_results))
        print("At the end, keeping ", len(all_results), " unique hits")
    
    return all_results
        
def mmseqs2_results(results, query_fasta, output_format, output_file, num_threads):
    """
    Run MMseqs2 search instead of BLAST for the given results.
    """

    print("Running MMseqs2 now...")
    sequences = {}
    with open(query_fasta, "r") as f:
        name = None
        seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    sequences[name] = "".join(seq)
                name = line
                seq = []
            else:
                seq.append(line)
        if name is not None:
            sequences[name] = "".join(seq)

    # Create a temporary directory to store files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write all query sequences to a FASTA file (database)
        db_fasta = os.path.join(tmpdir, "db.fasta")
        with open(db_fasta, "w") as dbf:
            for name, seq in sequences.items():
                dbf.write(f"{name}\n{seq}\n")

        # Write all results sequences to a FASTA file
        query_fasta = os.path.join(tmpdir, "target.fasta")
        with open(query_fasta, "w") as qf:
            for name, seq in results:
                qf.write(f">{"".join(name.split()[1:])}#{name.split()[0][1:]}\n{seq}\n")

        # Create MMseqs2 database
        db_mmseqs = os.path.join(tmpdir, "db_mmseqs")
        query_mmseqs = os.path.join(tmpdir, "query_mmseqs")
        result_mmseqs = os.path.join(tmpdir, "result_mmseqs")
        tmp_mmseqs = os.path.join(tmpdir, "tmp_mmseqs")

        subprocess.run(["mmseqs", "createdb", db_fasta, db_mmseqs], check=True)
        subprocess.run(["mmseqs", "createdb", query_fasta, query_mmseqs], check=True)

        print("mmseq created databases ")

        # Run MMseqs2 search
        subprocess.run([
            "mmseqs", "search", query_mmseqs, db_mmseqs, result_mmseqs, tmp_mmseqs,
            "--threads", str(num_threads)
        ], check=True)

        # Convert results to tabular format
        result_tsv = os.path.join(tmpdir, "result.tsv")
        subprocess.run([
            "mmseqs", "convertalis", query_mmseqs, db_mmseqs, result_mmseqs, result_tsv,
            "--format-mode", output_format
        ], check=True)

        # Output results
        with open(result_tsv, "r") as resf:
            if output_file:
                with open(output_file, "w") as out_f:
                    out_f.write(resf.read())
            else:
                print(resf.read())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Query a FAISS database with a sequence.")
    parser.add_argument("--database", required=True, help="Path to the folder containing FAISS database files.")
    parser.add_argument("--query_sequences", required=True, help="Fasta file of queris")
    parser.add_argument("--force_cpu", action="store_true", help="Force the use of CPU even if GPUs are available.")
    parser.add_argument("-o", "--output", required=False, help="Path to the output file [stdout]")
    parser.add_argument("--outfmt", type=str, default='0', help="Format of the BLAST output")
    parser.add_argument("-t", "--num_threads", type=int, default=1, help="Number of threads to use for parallel querying.")
    parser.add_argument("--subdatabases_size", type=int, default=10000000, help="Number of vectors in each faiss database")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()
    path_to_centroid_to_prots = os.path.join(os.path.dirname(__file__), "centroid_to_prots")

    if args.output and os.path.exists(args.output):
        os.remove(args.output)
    
    # Check if GPUs are available
    if not args.force_cpu :
        try:
            gpus_available = torch.cuda.is_available()
            print("GPU available, moving on")
        except ImportError:
            gpus_available = False
            print("No GPU available, using CPUs...this could be slow if you have many queries")
    else:
        gpus_available = False

    database = args.database.strip("/")

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

    t1 = time.time()
    query_results = query_faiss_database(
        original_fasta=database+"/centroids.fa",
        database_folder=database,
        query_sequences=query_sequences,
        query_names=query_names,
        gpus_available=gpus_available,
        cutoff = 0.2,
        subdatabase_size=args.subdatabases_size,
        max_workers=args.num_threads
    )
    t2 = time.time()
    print(f"Time for querying FAISS database: {t2 - t1:.2f} seconds. {len(query_results)} centroids")

    # Sort results by query index and then by ascending distance
    query_results.sort(key=lambda x: (x[0], x[3]))

    # Output query_results as an intermediate FASTA file
    if args.output:
        intermediate_fasta = args.output + ".intermediate.fasta"
    else:
        intermediate_fasta = "query_results_intermediate.fasta"
    with open(intermediate_fasta, "w") as fasta_file:
        for query_name, centroid_name, sequence, distance in query_results:
            fasta_file.write(f"{centroid_name}#{query_name}{sequence}\n")
    print(f"Intermediate FASTA file written: {intermediate_fasta}")

    tsv_output = args.output + ".tsv" if args.output else "query_results.tsv"
    with open(tsv_output, "w") as tsvfile:
        tsvfile.write("#query_name\tresult_name\tresult_sequences\tcosine_distance\n")
        for query_name, centroid_name, sequence, distance in query_results:
            tsvfile.write(f"{query_name.strip()}\t{centroid_name.strip()[1:].split()[0]}\t{sequence}\t{distance}\n")
    print(f"TSV file written: {tsv_output}")

    # Write a deduplicated intermediate FASTA file (unique centroid name/seq pairs)
    if args.output:
        unique_fasta = args.output + ".unique_centroids.fasta"
    else:
        unique_fasta = "unique_centroids.fasta"
    unique_centroids = set()
    for _, centroid_name, sequence, _ in query_results:
        unique_centroids.add((centroid_name, sequence))
    with open(unique_fasta, "w") as fasta_file:
        for centroid_name, sequence in unique_centroids:
            fasta_file.write(f"{centroid_name}\n{sequence}\n")
    print(f"Unique centroid FASTA file written: {unique_fasta}")
    
    # with open("query_results.pkl", "wb") as f:
    #     pickle.dump(query_results, f)
    # with open("query_results.pkl", "rb") as f:
    #     query_results = pickle.load(f)
    # print("query results loaded ", len(query_results))

    centroid_hits = list(set([x[1] for x in query_results]))
    all_results = obtain_all_proteins(centroid_hits, database+"/all_prots", path_to_centroid_to_prots, args.num_threads)
    t3= time.time()
    # print(f"Time for obtaining all proteins: {t3 - t2:.2f} seconds. {len(all_results)} proteins")

    with open("all_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("all_results pickle dumped")
    with open("all_results.pkl", "rb") as f:
        all_results = pickle.load(f)
    print("all_results pickle loaded, length:", len(all_results))

    if args.output:
        fasta_output = args.output + ".fasta"
    else:
        fasta_output = "results.fasta"

    with open(fasta_output, "w") as fasta_file:
        for name, seq in all_results:
            if '>' in seq:
                print("WHWUHIA ", seq)
            fasta_file.write(f">{"".join(name.split()[1:])}#{name.split()[0][1:]}\n{seq}\n")

    mmseqs2_results(all_results, args.query_sequences, args.outfmt, args.output+".mmseqs2", args.num_threads)
    t4 = time.time()

    command = "awk '!seen[$1]++' " + args.output+".mmseqs2" + " > " + args.output + ".top_hit" #to keep only the first hit
    subprocess.run(command, shell=True, check=True)

    print(f"Time for querying FAISS database: {t2 - t1:.2f} seconds")
    print(f"Time for obtaining all proteins: {t3 - t2:.2f} seconds")
    print(f"Time for running MMseqs2: {t4 - t3:.2f} seconds")
    print(f"Total time: {t4 - t1:.2f} seconds")


    # if args.output:
    #     with open(args.output, "w") as output_file:
    #         output_file.write("#query_name\thit_name\thit_sequence\n")
    #         for query, name, sequence in all_results:
    #             # Extract hit name (remove '>' if present)
    #             hit_name = name.strip().lstrip('>').split(' ')[0]
    #             query_name = query.strip().lstrip('>').split(' ')[0]
    #             output_file.write(f"{query_name}\t{hit_name}\t{sequence}\n")
    # else:
    #     # Print header
    #     print("#query_name\thit_name\thit_sequence")
    #     for query, name, sequence, in all_results:
    #         hit_name = name.strip().lstrip('>').split(' ')[0]
    #         query_name = query.strip().lstrip('>').split(' ')[0]
    #         print(f"{query_name}\t{hit_name}\t{sequence}")

