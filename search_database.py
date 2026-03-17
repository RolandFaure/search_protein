#!/usr/bin/env python3
"""
Search a FAISS database using pre-computed query embeddings.
"""

import faiss
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import numpy as np
import sys
import tempfile
import subprocess
from sklearn.metrics.pairwise import cosine_distances
from usearch.index import Index,search, MetricKind, BatchMatches

__version__ = "1.4.0"

def reduce_number_of_embeddings(query_embeddings, threshold):
    """
    Cluster embeddings using cosine distance and return representative embeddings.
    All embeddings within a cluster have cosine distance < 0.1 from the representative.
    Returns the representative embeddings and the maximum distance between any representative
    and its clustered vectors.
    """
    
    if len(query_embeddings) == 0:
        return query_embeddings, 0.0
    
    # Normalize embeddings for cosine distance calculation
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    normalized_embeddings = query_embeddings / (norms + 1e-10)
    
    # Greedy clustering
    representatives = []
    representative_indices = []
    clustered = np.zeros(len(query_embeddings), dtype=bool)
    max_distance = 0.0
    
    for i in range(len(query_embeddings)):
        if clustered[i]:
            continue
        
        # This embedding becomes a representative
        representatives.append(query_embeddings[i])
        representative_indices.append(i)
        clustered[i] = True
        
        # Find all unclustered embeddings within 0.1 cosine distance
        distances = cosine_distances(normalized_embeddings[i:i+1], normalized_embeddings)[0]
        within_threshold = (distances < threshold) & (~clustered)
        
        # Track maximum distance to any clustered vector
        if np.any(within_threshold):
            max_distance = max(max_distance, np.max(distances[within_threshold]))
        
        clustered[within_threshold] = True
    
    print(f"Reduced {len(query_embeddings)} embeddings to {len(representatives)} representatives (max distance: {max_distance:.4f})")
    return np.array(representatives), max_distance

def query_bin(bin_file, original_fasta, database_folder, query_embeddings, query_names, subdatabase_size, cutoff):
    start_time = time.time()
    faiss.omp_set_num_threads(1)

    file_starting_pos = int(bin_file.strip(".bin").split("_")[2])*subdatabase_size
    index_path = os.path.join(database_folder, bin_file)
    # Load the FAISS index using memory mapping to avoid loading entirely in RAM
    # index = faiss.read_index(index_path, faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY)
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
    with open(names_file, "rb") as nf, open(original_fasta, "r") as fastafile : #, open("/pasteur/appa/scratch/rfaure/human_db_usearch/centroids.fa.embeddings", "rb") as embeddings_file:
        for query_idx in range(len(results_sorted)):
            for result in results_sorted[query_idx]:
                nf.seek(8 * result[0])
                position_name = int.from_bytes(nf.read(8), byteorder='little', signed=False)
                fastafile.seek(position_name)
                name_line = fastafile.readline().strip()
                sequence_line = fastafile.readline().strip()
                final_results.append((query_names[query_idx], name_line, sequence_line, result[1]))

                # #debug: for each result, also get the embedding and actually compute the cosine distance to be more accurate (use the same way of computing it as in reduce_number_of_embeddings, so that the cutoff is consistent)
                # embeddings_file.seek(512 * 2 * result[0])  # each embedding is 512 floats, each float is 2 bytes
                # embedding_bytes = embeddings_file.read(512 * 2)
                # embedding = np.frombuffer(embedding_bytes, dtype=np.float16)
                # embedding = embedding.astype(np.float32)  # convert to float32 for distance calculation
                # query_embedding = query_embeddings[query_idx].reshape(1, -1)
                # embedding = embedding.reshape(1, -1)
                # # Normalize both embeddings
                # query_embedding_norm = np.linalg.norm(query_embedding)
                # embedding_norm = np.linalg.norm(embedding)
                # if embedding_norm > 0:
                #     query_embedding = query_embedding / embedding_norm
                #     embedding = embedding / embedding_norm
                # # Compute cosine distance and L2 distance to check the correlation between them
                # dot_product = np.dot(query_embedding, embedding.T)[0][0]
                # cosine_distance = 1 - dot_product
                # l2_distance = np.linalg.norm(query_embedding - embedding)
                # print(f"Distance to centroid {name_line} for query {query_names[query_idx]}: {cosine_distance:.4f} (original distance: {result[1]:.4f}), L2 distance: {l2_distance:.4f}")

    elapsed_time = time.time() - start_time
    print(f"Total matches for bin {bin_file}: {np.sum([len(result) for result in results])}")
    print(f"Time taken for querying bin {bin_file}: {elapsed_time:.2f} seconds")
    return final_results


def parallel_query_bins(original_fasta, database_folder, query_embeddings, query_names, cutoff, subdatabase_size=10000000, max_workers=4):
    faiss_database_folder = database_folder+"/faiss"
    bin_files = [file for file in os.listdir(faiss_database_folder) if file.endswith(".bin")]

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(query_bin, bin_file, original_fasta, faiss_database_folder, query_embeddings, query_names, subdatabase_size, cutoff) for bin_file in bin_files
        ]
        for future in as_completed(futures):
            all_results.extend(future.result())

    return all_results


def search_faiss_database(original_fasta, database_folder, query_embeddings, query_names, cutoff=0.2, subdatabase_size=10000000, max_workers=4):
    """
    Searches a FAISS database with pre-computed embeddings and retrieves the nearest neighbors in parallel.

    Args:
        original_fasta (str): Path to the original FASTA file.
        database_folder (str): Path to the folder containing FAISS database files (.bin).
        query_embeddings (np.ndarray): Pre-computed embeddings for query sequences.
        query_names (list of str): List of query names.
        cutoff (float): Distance cutoff for results.
        subdatabase_size (int): Number of vectors in each FAISS subdatabase.
        max_workers (int): Number of parallel workers.

    Returns:
        list of tuple: Each tuple contains (query_name, name_line, sequence_line, distance).
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Querying in parallel...")
    results = parallel_query_bins(
        original_fasta=original_fasta,
        database_folder=database_folder,
        query_embeddings=query_embeddings,
        query_names=query_names,
        cutoff=cutoff,
        subdatabase_size=subdatabase_size,
        max_workers=max_workers
    )

    return results


def query_usearch_bin(bin_file, original_fasta, database_folder, query_embeddings, query_names, subdatabase_size, cutoff):
    """Query a single usearch index file and return results."""
    start_time = time.time()

    file_starting_pos = int(bin_file.strip(".bin").split("_")[2]) * subdatabase_size
    index_path = os.path.join(database_folder, bin_file)
    
    # Load the usearch index
    index = Index(ndim=512, metric='cos')
    index.load(index_path)
    
    # Search with increasing k until we get results beyond the cutoff
    nb_of_searches = 1
    all_matches = None
    while nb_of_searches <= 10:
        k = 20 * 2 ** nb_of_searches
        # print("query embedding shape ", query_embeddings.shape, " searching with k=", k, " on index ", index_path)
        # query_embeddings = np.array([query_embeddings[0], query_embeddings[0]])
        matches : BatchMatches = index.search(query_embeddings, k, exact=True)
        if query_embeddings.shape[0] == 1:
            matches = [matches]

        # Check if we got far enough results
        max_distance = 0
        for i in range(len(matches)):
            if len(matches[i]) > 0:
                # Get the farthest match for this query
                last_valid_idx = len(matches[i]) - 1
                max_distance = max(max_distance, matches[i][last_valid_idx].distance)
        
        if max_distance >= cutoff or nb_of_searches >= 10:
            all_matches = matches
            break
        
        nb_of_searches += 1
    
    if all_matches is None:
        return []

    # print("Finished searching with k=", k, " max distance found: ", max_distance, " number of matches: ", sum(len(m) for m in all_matches))
    
    # Process results
    results = [[] for _ in range(len(query_embeddings))]
    for i in range(len(all_matches)):
        for j in range(len(all_matches[i])):
            dist = all_matches[i][j].distance
            if dist < cutoff:
                key = all_matches[i][j].key
                results[i].append((file_starting_pos + key, dist))

    print(f"Total matches for bin {bin_file}: {np.sum([len(result) for result in results])}")
    
    results_sorted = [sorted(result, key=lambda x: x[1]) for result in results]
    final_results = []
    names_file = os.path.join(database_folder, f"{os.path.basename(original_fasta)}.names")
    with open(names_file, "rb") as nf, open(original_fasta, "r") as fastafile : #, open("/pasteur/appa/scratch/rfaure/human_db_usearch/centroids.fa.embeddings", "rb") as embeddings_file:
        for query_idx in range(len(results_sorted)):
            for result in results_sorted[query_idx]:
                nf.seek(8 * int(result[0]))
                position_name = int.from_bytes(nf.read(8), byteorder='little', signed=False)
                fastafile.seek(position_name)
                name_line = fastafile.readline().strip()
                sequence_line = fastafile.readline().strip()
                final_results.append((query_names[query_idx], name_line, sequence_line, result[1]))

                # #now get the embedding and actually compute the cosine distance to be more accurate (use the same way of computing it as in reduce_number_of_embeddings, so that the cutoff is consistent)
                # embeddings_file.seek(512 * 2 * int(result[0]))  # each embedding is 512 floats, each float is 2 bytes
                # embedding_bytes = embeddings_file.read(512 * 2)
                # embedding = np.frombuffer(embedding_bytes, dtype=np.float16)
                # embedding = embedding.astype(np.float32)  # convert to float32 for distance calculation
                # query_embedding = query_embeddings[query_idx].reshape(1, -1)
                # embedding = embedding.reshape(1, -1)
                # # Normalize both embeddings                query_embedding_norm = np.linalg.norm(query_embedding)
                # embedding_norm = np.linalg.norm(embedding)
                # if embedding_norm > 0:
                #     query_embedding = query_embedding / embedding_norm
                #     embedding = embedding / embedding_norm
                # # Compute cosine distance
                # dot_product = np.dot(query_embedding, embedding.T)[0][0]
                # cosine_distance = 1 - dot_product
                # print(f"Distance to centroid {name_line} for query {query_names[query_idx]}: {cosine_distance:.4f} (original distance: {result[1]:.4f})")
                # #grep the name in /pasteur/appa/scratch/rfaure/queries/petases/output_human_one_petase/intermediate_files/query_results_intermediate.fasta and exit if it is not found, to check that we are actually looking at the same results
                # with open("/pasteur/appa/scratch/rfaure/queries/petases/output_human_one_petase/intermediate_files/query_results_intermediate.fasta", "r") as intermediate_fasta:
                #     if os.system(f"grep '{name_line}' /pasteur/appa/scratch/rfaure/queries/petases/output_human_one_petase/intermediate_files/query_results_intermediate.fasta > /dev/null") != 0:
                #         print(f"ERROR: {name_line} not found in intermediate FASTA, something is wrong!")
                #         sys.exit(1)

    elapsed_time = time.time() - start_time
    print(f"Time taken for querying bin {bin_file}: {elapsed_time:.2f} seconds")
    return final_results


def parallel_query_usearch_bins(original_fasta, database_folder, query_embeddings, query_names, cutoff, subdatabase_size=10000000, max_workers=4):
    """Query multiple usearch index files in parallel."""
    usearch_database_folder = database_folder + "/usearch"
    bin_files = [file for file in os.listdir(usearch_database_folder) if file.endswith(".bin")]

    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(query_usearch_bin, bin_file, original_fasta, usearch_database_folder, query_embeddings, query_names, subdatabase_size, cutoff) 
            for bin_file in bin_files
        ]
        for future in as_completed(futures):
            all_results.extend(future.result())
    return all_results


def search_usearch_database(original_fasta, database_folder, query_embeddings, query_names, cutoff=0.2, subdatabase_size=10000000, max_workers=4):
    """
    Searches a usearch database with pre-computed embeddings and retrieves the nearest neighbors in parallel.

    Args:
        original_fasta (str): Path to the original FASTA file.
        database_folder (str): Path to the folder containing usearch database files (.bin).
        query_embeddings (np.ndarray): Pre-computed embeddings for query sequences.
        query_names (list of str): List of query names.
        cutoff (float): Distance cutoff for results.
        subdatabase_size (int): Number of vectors in each usearch subdatabase.
        max_workers (int): Number of parallel workers.

    Returns:
        list of tuple: Each tuple contains (query_name, name_line, sequence_line, distance).
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Querying usearch database in parallel...")
    results = parallel_query_usearch_bins(
        original_fasta=original_fasta,
        database_folder=database_folder,
        query_embeddings=query_embeddings,
        query_names=query_names,
        cutoff=cutoff,
        subdatabase_size=subdatabase_size,
        max_workers=max_workers
    )

    return results


def _process_centroid_result(args_tuple):
    """
    Helper function for obtain_all_proteins. Must be at module level for ProcessPoolExecutor.
    Extracts proteins for a single centroid.
    """
    centroid_name, database_all_proteins, path_to_centroid_to_prots = args_tuple
    centroid_id = centroid_name.strip().lstrip('>').split(' ')[0]
    proteins = []
    tmp_filename = None
    try:
        fd, tmp_filename = tempfile.mkstemp(prefix=f"tmp_{os.getpid()}_", suffix=".fa")
        os.close(fd)

        with open(tmp_filename, "w") as tmp_out:
            subprocess.run(
                [path_to_centroid_to_prots, database_all_proteins, centroid_id],
                stdout=tmp_out,
                check=True,
            )

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

        return proteins
    finally:
        if tmp_filename and os.path.exists(tmp_filename):
            os.remove(tmp_filename)


def obtain_all_proteins(centroids, database_all_proteins, path_to_centroid_to_prots, num_threads):
    """
    For each result, runs an external command to extract proteins and parses the output.
    """
    all_results = []
    # Use ProcessPoolExecutor instead of ThreadPoolExecutor to avoid GIL contention
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        # Pass all arguments as a tuple since process_result needs to be at module level
        futures = [executor.submit(_process_centroid_result, (centroid, database_all_proteins, path_to_centroid_to_prots)) for centroid in centroids]
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
                #there are two different formats
                if len("".join("_".join(name.split("_")[:3]).split())) < len(name.split(" ")[0]) :
                    processed_name = "".join("_".join(name.split("_")[3:]).split())
                else :
                    processed_name = "".join(name.split(" ")[1:])
                qf.write(f">{processed_name}\n{seq}\n")

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
                    if output_format == '0':  # this is the default, output the header
                        out_f.write("#query\ttarget\tidentity\talignment_length\tnb_mismatches\tnb_gap_openings\tquery_start\tquery_end\ttarget_start\ttarget_end\tevalue\tbitscore\n")
                    out_f.write(resf.read())
            else:
                print(resf.read())

        # Parse MMseqs2 tabular results to collect matched query IDs and write a FASTA
        matched_queries = set()
        with open(result_tsv, "r") as resf:
            for line in resf:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) >= 1:
                    matched_queries.add(cols[0].lstrip(">"))

        # Read the query (target) FASTA we created and keep only matched entries
        matched_records = []
        with open(query_fasta, "r") as qf:
            header = None
            seq_lines = []
            for line in qf:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if header is not None:
                        h_full = header.lstrip(">")
                        h_first = h_full.split()[0]
                        if h_full in matched_queries or h_first in matched_queries:
                            matched_records.append((header, "".join(seq_lines)))
                    header = line
                    seq_lines = []
                else:
                    seq_lines.append(line.strip())
            if header is not None:
                h_full = header.lstrip(">")
                h_first = h_full.split()[0]
                if h_full in matched_queries or h_first in matched_queries:
                    matched_records.append((header, "".join(seq_lines)))

        # Write matched proteins to a FASTA file in the output folder root
        if output_file:
            # Extract the output folder from the output_file path
            output_folder = os.path.dirname(output_file)
            matched_fasta = os.path.join(output_folder, "matches.fasta")
        else:
            matched_fasta = os.path.join(tmpdir, "matched_proteins.fasta")

        with open(matched_fasta, "w") as mf:
            for header, seq in matched_records:
                mf.write(f"{header}\n{seq}\n")

        print(f"Wrote {len(matched_records)} matched proteins to {matched_fasta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search a database (FAISS or Milvus) using pre-computed embeddings.")
    parser.add_argument("--database", required=True, help="Path to the folder containing database files.")
    parser.add_argument("--output", "-o", required=True, help="Path to the output folder (created by embed_query.py)")
    parser.add_argument("--query_sequences", required=True, help="Fasta file of queries (for MMseqs2 step)")
    parser.add_argument("--db-type", type=str, choices=['faiss', 'usearch'], default='faiss', help="Database type to use: faiss or usearch (default: faiss)")
    parser.add_argument("--outfmt", type=str, default='0', help="Format of the output [1: SAM]")
    parser.add_argument("-t", "--num_threads", type=int, default=1, help="Number of threads to use for parallel querying.")
    #parser.add_argument("--subdatabases_size", type=int, default=10000000, help="Number of vectors in each faiss database")
    #parser.add_argument("--cutoff", type=float, default=0.2, help="Distance cutoff for results")
    #parser.add_argument("-r","--reduce_embeddings", action="store_true", help="Cluster similar embeddings to reduce search time")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()
    cutoff = 0.2 #cosine distance cutoff
    subdatabase_size = 10000000
    reduce_embeddings = True
    # reduce_embeddings = False
    # print("DEBUG: reduce_embeddings is set to ", reduce_embeddings )
    group_distance = 0.1
    path_to_centroid_to_prots = os.path.join(os.path.dirname(__file__), "centroid_to_prots")


    # Set up folder structure
    output_folder = args.output.rstrip("/")
    if not os.path.exists(output_folder):
        print(f"Error: Output folder '{output_folder}' does not exist. Run embed_query.py first.")
        exit(1)
    
    intermediate_folder = os.path.join(output_folder, "intermediate_files")
    if not os.path.exists(intermediate_folder):
        print(f"Error: intermediate_files folder not found in '{output_folder}'. Run embed_query.py first.")
        exit(1)

    database = args.database.rstrip("/")

    # Load embeddings from intermediate folder
    embeddings_file = os.path.join(intermediate_folder, "query_embeddings.npy")
    print(f"Loading embeddings from {embeddings_file}")
    if not os.path.exists(embeddings_file):
        print(f"Error: Embeddings file not found at {embeddings_file}")
        exit(1)
    query_embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings with shape {query_embeddings.shape}")

    #check that the embeddings are normalized, if not normalize them and print a warning
    norms = np.linalg.norm(query_embeddings, axis=1)
    if not np.allclose(norms, 1, atol=1e-3):
        print("Warning: Embeddings are not normalized, normalizing now...")
        query_embeddings = query_embeddings / (norms[:, np.newaxis] + 1e-10)

    # Load query names from intermediate folder
    names_file = os.path.join(intermediate_folder, "query_embeddings.names.txt")
    query_names = []
    if os.path.exists(names_file):
        print(f"Loading query names from {names_file}")
        with open(names_file, 'r') as f:
            query_names = [line.strip() for line in f]
    else:
        print(f"Warning: Query names file not found at {names_file}")
        print("Generating generic query names...")
        query_names = [f"query_{i}" for i in range(len(query_embeddings))]

    print(f"Number of queries: {len(query_names)}")

    # Optionally reduce embeddings by clustering similar ones
    if reduce_embeddings:
        print("Reducing embeddings by clustering...")
        original_count = len(query_embeddings)
        query_embeddings, max_dist = reduce_number_of_embeddings(query_embeddings, threshold=group_distance)
        # For reduced embeddings, we use generic names since we're representing clusters
        query_names = [f"cluster_{i}" for i in range(len(query_embeddings))]
        print(f"Embeddings reduced from {original_count} to {len(query_embeddings)}")
        cutoff += max_dist

    t1 = time.time()
    
    # Choose database type
    if args.db_type == 'usearch':
        query_results = search_usearch_database(
            original_fasta=database+"/centroids.fa",
            database_folder=database,
            query_embeddings=query_embeddings,
            query_names=query_names,
            cutoff=cutoff,
            subdatabase_size=subdatabase_size,
            max_workers=args.num_threads
        )
        db_name = "usearch"
    else:  # faiss
        cutoff = cutoff * 2  # convert cosine distance to L2² (FAISS Flat index returns squared L2 distance)
        print("the new cutoff for FAISS search is ", cutoff)
        query_results = search_faiss_database(
            original_fasta=database+"/centroids.fa",
            database_folder=database,
            query_embeddings=query_embeddings,
            query_names=query_names,
            cutoff=cutoff,
            subdatabase_size=subdatabase_size,
            max_workers=args.num_threads
        )
        db_name = "FAISS"
    
    t2 = time.time()
    print(f"Time for querying {db_name} database: {t2 - t1:.2f} seconds. {len(query_results)} centroids")

    if len(query_results) == 0:
        print("No results, exiting")
        print(f"Total time: {t2 - t1:.2f} seconds")
        sys.exit(0)

    # Sort results by query index and then by ascending distance
    query_results.sort(key=lambda x: (x[0], x[3]))

    # Output query_results as an intermediate FASTA file in intermediate_files
    intermediate_fasta = os.path.join(intermediate_folder, "query_results_intermediate.fasta")
    with open(intermediate_fasta, "w") as fasta_file:
        for query_name, centroid_name, sequence, distance in query_results:
            fasta_file.write(f"{centroid_name}#{query_name.strip()}\n{sequence}\n")
    # print(f"Intermediate FASTA file written: {intermediate_fasta}")
    # print("EXITITNG NOW TO CHECK INTERMEDIATE FILES, COMMENT THIS EXIT TO RUN THE WHOLE PIPELINE")
    # sys.exit(0)

    # Output TSV file in intermediate_files
    tsv_output = os.path.join(intermediate_folder, "query_results.tsv")
    with open(tsv_output, "w") as tsvfile:
        tsvfile.write("#query_name\tresult_name\tresult_sequences\tcosine_distance\n")
        for query_name, centroid_name, sequence, distance in query_results:
            tsvfile.write(f"{query_name.strip()}\t{centroid_name.strip()[1:].split()[0]}\t{sequence}\t{distance}\n")
    print(f"TSV file written: {tsv_output}")

    # Write a deduplicated intermediate FASTA file in intermediate_files (unique centroid name/seq pairs)
    unique_fasta = os.path.join(intermediate_folder, "unique_centroids.fasta")
    unique_centroids = set()
    for _, centroid_name, sequence, _ in query_results:
        unique_centroids.add((centroid_name, sequence))
    with open(unique_fasta, "w") as fasta_file:
        for centroid_name, sequence in unique_centroids:
            fasta_file.write(f"{centroid_name}\n{sequence}\n")
    print(f"Unique centroid FASTA file written: {unique_fasta}")

    centroid_hits = list(set([x[1] for x in query_results]))
    all_results = obtain_all_proteins(centroid_hits, database+"/all_prots", path_to_centroid_to_prots, args.num_threads)
    t3 = time.time()

    # Write all results to intermediate_files
    fasta_output = os.path.join(intermediate_folder, "all_results.fasta")
    with open(fasta_output, "w") as fasta_file:
        for name, seq in all_results:
            if '>' in seq:
                print("WARNING{db_name} found in sequence: ", seq)
            if len(name.split()[0].split("_")) < 4:  # that's because the human and nonhuman db are not exactly formatted the same way
                header_parts = ''.join(name.split()[1:])
                accession = name.split()[0][1:]
                fasta_file.write(f">{header_parts}#{accession}\n{seq}\n")
            else:
                # Extract header and accession from the name
                header_parts = "_".join(name.split("_")[3:])
                header_clean = "".join(header_parts.split())
                accession = "_".join(name.lstrip('>').split("_")[:3])
                fasta_file.write(f">{header_clean}#{accession}\n{seq}\n")

    # Run MMseqs2 and write main output files to output folder root
    mmseqs2_output = os.path.join(output_folder, "matches.mmseqs2")
    mmseqs2_results(all_results, args.query_sequences, args.outfmt, mmseqs2_output, args.num_threads)
    t4 = time.time()

    # Create top hit file in intermediate_files
    top_hit_file = os.path.join(intermediate_folder, "matches.top_hit")
    command = f"awk '!seen[$1]++' {mmseqs2_output} > {top_hit_file}"  # to keep only the first hit
    subprocess.run(command, shell=True, check=True)

    print(f"Time for querying FAISS database: {t2 - t1:.2f} seconds")
    print(f"Time for obtaining all proteins: {t3 - t2:.2f} seconds")
    print(f"Time for running MMseqs2: {t4 - t3:.2f} seconds")
    print(f"Total time: {t4 - t1:.2f} seconds")
    
