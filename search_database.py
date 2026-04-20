#!/usr/bin/env python3
"""
Search a FAISS/usearch database using pre-computed query embeddings.
"""
import os
#limit the number of threads used by FAISS and other libraries to avoid oversubscription when using ProcessPoolExecutor
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import faiss
import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import numpy as np
import sys
import tempfile
import subprocess
from sklearn.metrics.pairwise import cosine_distances
from usearch.index import Index,search, MetricKind, BatchMatches
import datetime
import pca

__version__ = "2.3.0"

def load_pca_if_exists(database_folder):
    """
    Load PCA model and components from database folder if it exists.
    
    Args:
        database_folder (str): Path to the database folder.
    
    Returns:
        tuple: (PCA model, n_components) or (None, None) if PCA doesn't exist
    """
    pca_folder = os.path.join(database_folder, "pca")
    pca_model_file = os.path.join(pca_folder, "pca_model.pkl")
    
    if not os.path.exists(pca_model_file):
        return None, None
    
    import pickle
    with open(pca_model_file, "rb") as f:
        pca = pickle.load(f)
    
    n_components = pca.n_components_
    return pca, n_components

def apply_pca_to_query(query_embeddings, pca):
    """
    Apply PCA transformation to query embeddings and keep only first n_components.
    
    Args:
        query_embeddings (np.ndarray): Query embeddings (shape: [n_queries, 512])
        pca: Fitted PCA model
    
    Returns:
        np.ndarray: Transformed embeddings (shape: [n_queries, n_components])
    """
    transformed = pca.transform(query_embeddings.astype(np.float32))
    # Normalize after PCA to preserve unit length property
    transformed = transformed.astype(np.float32)
    norms = np.linalg.norm(transformed, axis=1, keepdims=True)
    transformed = transformed / (norms + 1e-10)
    return transformed

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
                final_results.append((query_names[query_idx], name_line, sequence_line, result[1], int(result[0])))

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
    index.load(index_path) #without memory-mapping
    # index = Index.restore(index_path, view=True) #with memory-mapping
    
    # # Load PCA model if it exists (needed for debug distance calculations)
    # pca, _ = load_pca_if_exists("/pasteur/appa/scratch/rfaure/nonhuman_db_usearch_pca/")
    # print("DEBUG osfuoiu")
    
    # Search with increasing k until we get results beyond the cutoff
    nb_of_searches = 1
    all_matches = None
    while nb_of_searches <= 10:
        k = 2000 * 2 ** nb_of_searches
        # print("query embedding shape ", query_embeddings.shape, " searching with k=", k, " on index ", index_path)
        # query_embeddings = np.array([query_embeddings[0], query_embeddings[0]])
        matches : BatchMatches = index.search(query_embeddings, k, exact=False)
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
    time_usearch = time.time()
    
    results_sorted = [sorted(result, key=lambda x: x[1]) for result in results]
    final_results = []
    names_file = os.path.join(database_folder, f"{os.path.basename(original_fasta)}.names")
    with open(names_file, "rb") as nf, open(original_fasta, "r") as fastafile:
        for query_idx in range(len(results_sorted)):
            for result in results_sorted[query_idx]:
                nf.seek(8 * int(result[0]))
                position_name = int.from_bytes(nf.read(8), byteorder='little', signed=False)
                fastafile.seek(position_name)
                name_line = fastafile.readline().strip()
                sequence_line = fastafile.readline().strip()
                final_results.append((query_names[query_idx], name_line, sequence_line, result[1], int(result[0])))

    elapsed_time = time.time() - start_time
    print(f"Time taken for querying bin {bin_file}: {elapsed_time:.2f} seconds, time for post-processing: {time.time() - time_usearch:.2f} seconds")
    return final_results


def parallel_query_usearch_bins(original_fasta, database_folder, query_embeddings, query_names, cutoff, subdatabase_size=10000000, max_workers=4):
    """Query multiple usearch index files in parallel."""
    usearch_database_folder = database_folder + "/usearch"
    bin_files = [file for file in os.listdir(usearch_database_folder) if file.endswith(".bin")]
    # bin_files = bin_files[:100] # DEBUG - limit to first 10 bins for testing
    # print("DEBUUUUUGUGkkk")

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


def _load_embeddings_chunk(args_tuple):
    """
    Worker function to load a chunk of embeddings from disk.
    """
    embeddings_file_path, chunk_idxs = args_tuple
    
    chunk_embeddings = np.zeros((len(chunk_idxs), 512), dtype=np.float32)
    
    with open(embeddings_file_path, "rb") as embeddings_file:
        for i, c_idx in enumerate(chunk_idxs):
            embeddings_file.seek(512 * 2 * c_idx)
            embedding_bytes = embeddings_file.read(512 * 2)
            chunk_embeddings[i] = np.frombuffer(embedding_bytes, dtype=np.float16).astype(np.float32)
            
    return chunk_embeddings

def filter_pca_hits(query_embeddings_full, query_names, query_results, embeddings_file_path, cutoff, max_workers=4):
    """
    Filter hits returned by the PCA index by recalculating true Cosine distance 
    on the full 512-dimension vectors.
    """
    print("Filtering PCA hits using full 512-dimension embeddings to remove false positives...")
    
    if len(query_results) == 0:
        return query_results
        
    if not os.path.exists(embeddings_file_path):
        print(f"Warning: embeddings file not found at {embeddings_file_path}, skipping filtering.")
        return [res[:4] for res in query_results]

    time_start = time.time()

    # Normalize query embeddings
    norms = np.linalg.norm(query_embeddings_full, axis=1, keepdims=True)
    query_embeddings_full = query_embeddings_full / (norms + 1e-10)
    
    # Map query names to their index
    query_name_to_idx = {name: i for i, name in enumerate(query_names)}
    
    # Collect unique centroid indices to read to avoid duplicate reads
    unique_c_idxs = sorted(list(set([res[4] for res in query_results])))
    c_idx_to_matrix_idx = {c_idx: i for i, c_idx in enumerate(unique_c_idxs)}
    
    print(f"Loading {len(unique_c_idxs)} unique 512-dimension vectors across {max_workers} processes...")
    
    # Split the unique indices into chunks for multiprocessing
    chunk_size = max(1, len(unique_c_idxs) // max_workers)
    chunks = [unique_c_idxs[i:i + chunk_size] for i in range(0, len(unique_c_idxs), chunk_size)]
    
    # Prepare arguments for multiprocessing
    args_list = [(embeddings_file_path, chunk) for chunk in chunks]
    
    db_embeddings_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_load_embeddings_chunk, args): i for i, args in enumerate(args_list)}
        
        results = [None] * len(chunks)
        completed = 0
        for future in as_completed(futures):
            chunk_index = futures[future]
            results[chunk_index] = future.result()
            completed += 1
            print(f"Loaded {completed}/{len(chunks)} chunks for PCA filtering...", end="\r")
            
    print() # clear the carriage return
    
    # Concatenate all chunks sequentially
    db_embeddings = np.concatenate(results, axis=0)
            
    # Normalize DB embeddings matrix using vectorized operations
    norms_db = np.linalg.norm(db_embeddings, axis=1, keepdims=True)
    db_embeddings = db_embeddings / (norms_db + 1e-10)
    
    # Prepare indices for fast vectorized distance computation
    q_indices = np.array([query_name_to_idx[res[0]] for res in query_results])
    v_indices = np.array([c_idx_to_matrix_idx[res[4]] for res in query_results])
    
    Q_sub = query_embeddings_full[q_indices]
    V_sub = db_embeddings[v_indices]
    
    # Fast element-wise multiplication and sum across dimensions is mathematically exactly dot products matching Q_sub and V_sub
    dot_products = np.sum(Q_sub * V_sub, axis=1)
    cosine_distances = 1.0 - dot_products
    
    filtered_results = []
    for i, cosine_distance in enumerate(cosine_distances):
        if cosine_distance < cutoff:
            q_name, c_name, seq, pca_dist, c_idx = query_results[i]
            filtered_results.append((q_name, c_name, seq, cosine_distance))

    print(f"Filtered results from {len(query_results)} to {len(filtered_results)} valid hits in {time.time() - time_start:.2f} seconds.")
    return filtered_results


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

        try:
            with open(tmp_filename, "w") as tmp_out:
                subprocess.run(
                    [path_to_centroid_to_prots, database_all_proteins, centroid_id],
                    stdout=tmp_out,
                    check=True,
                )
        except subprocess.CalledProcessError:
            # Silently fail - return empty proteins list
            return proteins

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


def mmseqs2_results(results, query_fasta, output_format, output_file, num_threads, intermediate_folder=None):
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

        # Set up log files for mmseqs2 output
        log_file_1 = os.path.join(intermediate_folder, "mmseqs_createdb_1.log") if intermediate_folder else os.devnull
        log_file_2 = os.path.join(intermediate_folder, "mmseqs_createdb_2.log") if intermediate_folder else os.devnull
        log_file_search = os.path.join(intermediate_folder, "mmseqs_search.log") if intermediate_folder else os.devnull
        log_file_convertalis = os.path.join(intermediate_folder, "mmseqs_convertalis.log") if intermediate_folder else os.devnull
        
        with open(log_file_1, "w") as lf1, open(log_file_2, "w") as lf2, open(log_file_search, "w") as lfs, open(log_file_convertalis, "w") as lfc:
            subprocess.run(["mmseqs", "createdb", db_fasta, db_mmseqs], check=True, stdout=lf1, stderr=subprocess.STDOUT)
            subprocess.run(["mmseqs", "createdb", query_fasta, query_mmseqs], check=True, stdout=lf2, stderr=subprocess.STDOUT)

        print("mmseq created databases ")

        # Run MMseqs2 search
        with open(log_file_search, "w") as lfs:
            subprocess.run([
                "mmseqs", "search", query_mmseqs, db_mmseqs, result_mmseqs, tmp_mmseqs,
                "--threads", str(num_threads)
            ], check=True, stdout=lfs, stderr=subprocess.STDOUT)

        # Convert results to tabular format
        result_tsv = os.path.join(tmpdir, "result.tsv")
        with open(log_file_convertalis, "w") as lfc:
            subprocess.run([
                "mmseqs", "convertalis", query_mmseqs, db_mmseqs, result_mmseqs, result_tsv,
                "--format-mode", output_format
            ], check=True, stdout=lfc, stderr=subprocess.STDOUT)

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


def align_centroids_with_mmseqs2(unique_fasta, query_fasta, num_threads, intermediate_folder=None):
    """
    Align centroids FASTA file against query sequences using MMseqs2.
    Returns the set of centroid IDs (headers) that have at least one match.
    """
    print("Aligning centroids against query using MMseqs2...")
    
    # Create a temporary directory to store MMseqs2 files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Load query sequences
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

        # Create MMseqs2 database for queries (the database to search against)
        query_db_fasta = os.path.join(tmpdir, "query_db.fasta")
        with open(query_db_fasta, "w") as f:
            for name, seq in sequences.items():
                f.write(f"{name}\n{seq}\n")

        # Create MMseqs2 databases
        centroid_mmseqs = os.path.join(tmpdir, "centroid_mmseqs")
        query_mmseqs = os.path.join(tmpdir, "query_mmseqs")
        result_mmseqs = os.path.join(tmpdir, "result_mmseqs")
        tmp_mmseqs = os.path.join(tmpdir, "tmp_mmseqs")

        # Set up log files for mmseqs2 output
        log_file_1 = os.path.join(intermediate_folder, "mmseqs_centroid_createdb_1.log") if intermediate_folder else os.devnull
        log_file_2 = os.path.join(intermediate_folder, "mmseqs_centroid_createdb_2.log") if intermediate_folder else os.devnull
        log_file_search = os.path.join(intermediate_folder, "mmseqs_centroid_search.log") if intermediate_folder else os.devnull
        log_file_convertalis = os.path.join(intermediate_folder, "mmseqs_centroid_convertalis.log") if intermediate_folder else os.devnull
        
        with open(log_file_1, "w") as lf1, open(log_file_2, "w") as lf2, open(log_file_search, "w") as lfs, open(log_file_convertalis, "w") as lfc:
            subprocess.run(["mmseqs", "createdb", unique_fasta, centroid_mmseqs], check=True, stdout=lf1, stderr=subprocess.STDOUT)
            subprocess.run(["mmseqs", "createdb", query_db_fasta, query_mmseqs], check=True, stdout=lf2, stderr=subprocess.STDOUT)

        print("MMseqs2 databases created for centroid alignment")

        # Run MMseqs2 search: search centroids against query sequences
        with open(log_file_search, "w") as lfs:
            subprocess.run([
                "mmseqs", "search", centroid_mmseqs, query_mmseqs, result_mmseqs, tmp_mmseqs,
                "--threads", str(num_threads)
            ], check=True, stdout=lfs, stderr=subprocess.STDOUT)

        # Convert results to tabular formatG
        result_tsv = os.path.join(tmpdir, "result.tsv")
        with open(log_file_convertalis, "w") as lfc:
            subprocess.run([
                "mmseqs", "convertalis", centroid_mmseqs, query_mmseqs, result_mmseqs, result_tsv,
                "--format-mode", "0"
            ], check=True, stdout=lfc, stderr=subprocess.STDOUT)

        # Parse results to get matched centroids (unique query identifiers)
        matched_centroids = set()
        with open(result_tsv, "r") as resf:
            for line in resf:
                fields = line.strip().split("\t")
                if len(fields) > 0:
                    centroid_id = fields[0]  # First column is the centroid ID
                    matched_centroids.add(centroid_id)

        print(f"Found {len(matched_centroids)} centroids with at least one alignment to query")
        return matched_centroids


def calculate_index_threads(database_folder, db_type, max_threads, max_memory_gb):
    """
    Calculate the number of threads to use for index querying based on available memory.
    
    Strategy:
    - Look at the *_0.bin file size in the database
    - Estimate RAM needed per thread as: bin_file_size * 2
    - Deduce how many threads can run based on max_memory_gb
    - Return min(calculated_threads, max_threads)
    
    Args:
        database_folder (str): Path to the database folder
        db_type (str): 'faiss' or 'usearch'
        max_threads (int): Maximum number of threads available
        max_memory_gb (float): Maximum memory available in GB
    
    Returns:
        int: Number of threads to use for index querying
    """
    # Determine the correct subdirectory
    if db_type == 'usearch':
        index_dir = os.path.join(database_folder, "usearch")
    else:
        index_dir = os.path.join(database_folder, "faiss")
    
    # Find the _0.bin file
    bin_files = [f for f in os.listdir(index_dir) if f.endswith("_0.bin")]
    
    if not bin_files:
        print(f"Warning: No *_0.bin file found in {index_dir}. Defaulting to 1 thread for index querying.")
        return 1
    
    bin_path = os.path.join(index_dir, bin_files[0])
    bin_size_bytes = os.path.getsize(bin_path)
    bin_size_gb = bin_size_bytes / (1024 ** 3)
    
    # RAM needed per thread is approximately: bin_size * 2
    ram_per_thread_gb = bin_size_gb * 2
    
    if ram_per_thread_gb <= 0:
        print("Warning: Bin file is empty or too small. Defaulting to 1 thread.")
        return 1
    
    # Calculate how many threads we can run with available memory
    threads_by_memory = int(max_memory_gb / ram_per_thread_gb)
    
    # Use the minimum of threads limited by memory or max_threads
    index_threads = max(1, min(threads_by_memory, max_threads))
    
    print(f"Using {index_threads} threads for index querying based on the available memory and bin file size.")
    
    return index_threads


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search a database (FAISS or Usearch) using pre-computed embeddings. Can optionally embed query sequences on-the-fly.")
    parser.add_argument("--query_sequences", required=True, help="Fasta file of queries")
    parser.add_argument("--database", required=True, help="Path to the folder containing database files.")
    parser.add_argument("--output", "-o", required=True, help="Path to the output folder (created by embed_query.py if embedding step was run separately).")
    parser.add_argument("--db-type", type=str, choices=['faiss', 'usearch'], default='faiss', help="Database type to use: faiss or usearch (default: faiss)")
    parser.add_argument("--outfmt", type=str, default='0', help="Format of the mmseqs2 output [0], default is 0 which is a tabular format with header. See mmseqs2 documentation for details.")
    parser.add_argument("-m", "--memory", type=float, required=True, help="Maximum memory available in GB (mandatory)")
    parser.add_argument("-t", "--num_threads", type=int, required=True, help="Maximum number of threads available (mandatory)")
    parser.add_argument("--force_cpu", action="store_true", help="Force the use of CPU even if GPUs are available (for embedding step).")
    parser.add_argument("--deep-search", action="store_true", help="If enabled, extract proteins from all search results instead of only aligned centroids, then align everything with MMseqs2")
    #parser.add_argument("--subdatabases_size", type=int, default=10000000, help="Number of vectors in each faiss database")
    #parser.add_argument("--cutoff", type=float, default=0.2, help="Distance cutoff for results")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("search_database.py version ", __version__)
    print("command line used:\n", " ".join(sys.argv))
    
    # Set up folder structure
    output_folder = args.output.rstrip("/")
    intermediate_folder = os.path.join(output_folder, "intermediate_files")
    embeddings_file = os.path.join(intermediate_folder, "query_embeddings.npy")
    
    # Check if embeddings need to be created
    embeddings_exist = os.path.exists(embeddings_file)
    
    if not embeddings_exist:
        # Embeddings don't exist - we need to create them
        if not args.query_sequences:
            print("Error: Embeddings not found in output folder, and no --query_sequences provided.")
            print("Please provide --query_sequences, or run embed_query.py first and provide the folder.")
            exit(1)
        
        print(f"Embeddings not found in {output_folder}. Running embed_query.py automatically...")
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        
        # Import and run embed_query.py logic
        import sys as sys_module
        sys_module.path.insert(0, os.path.dirname(__file__))
        from embed_query import embed_query_sequences
        import torch
        import shutil
        
        # Check if GPU is available (unless forced to CPU)
        if not args.force_cpu:
            try:
                gpus_available = torch.cuda.is_available()
                if gpus_available:
                    print("GPU available for embedding")
                else:
                    print("No GPU available, using CPU for embedding")
            except ImportError:
                gpus_available = False
                print("No GPU available, using CPU for embedding")
        else:
            gpus_available = False
            print("Forced CPU mode for embedding")
        
        # Read query sequences from FASTA file
        query_sequences = []
        query_names = []
        sequence_now = ""
        
        print(f"Reading query sequences from {args.query_sequences}")
        with open(args.query_sequences, "r") as query_file_s:
            for line in query_file_s:
                if line.startswith(">"):
                    if sequence_now:
                        query_sequences.append(sequence_now)
                        sequence_now = ""
                    query_names.append(line.strip().lstrip('>'))
                else:
                    sequence_now += line.strip()
            if sequence_now:
                query_sequences.append(sequence_now)
        
        print(f"Found {len(query_sequences)} query sequences")
        
        # Create intermediate_files folder
        os.makedirs(intermediate_folder, exist_ok=True)
        
        # Embed the sequences
        batch_size = 10
        query_embeddings = embed_query_sequences(
            query_sequences=query_sequences,
            gpus_available=gpus_available,
            batch_size=batch_size
        )
        
        # Save embeddings
        np.save(embeddings_file, query_embeddings)
        print(f"Embeddings saved to {embeddings_file}")
        
        # Save query names
        names_file = os.path.join(intermediate_folder, "query_embeddings.names.txt")
        with open(names_file, 'w') as f:
            for name in query_names:
                f.write(f"{name}\n")
        print(f"Query names saved to {names_file}")
    else:
        print(f"Found existing embeddings in {output_folder}")
    
    # Validate that query_sequences is provided for MMseqs2 alignment
    if not args.query_sequences:
        print("Error: --query_sequences is required for MMseqs2 alignment step.")
        print("Please provide the path to your query FASTA file.")
        exit(1)
    
    # Validate output folder and intermediate files exist
    if not os.path.exists(output_folder):
        print(f"Error: Output '{output_folder}' does not exist.")
        exit(1)
    
    if not os.path.exists(intermediate_folder):
        print(f"Error: intermediate_files folder not found in '{output_folder}'.")
        exit(1)
    
    # Auto-calculate thread counts based on memory and available threads
    args.index_threads = calculate_index_threads(args.database, args.db_type, args.num_threads, args.memory)
    args.align_threads = args.num_threads  # Use all available threads for alignment
    
    print(f"Configuration: index_threads={args.index_threads}, align_threads={args.align_threads}")
    
    cutoff = 0.25 #cosine distance cutoff
    subdatabase_size = 10_000_000
    group_distance = 0 #when using 0.1, the papilloma query hit 10B proteins, which is too much
    path_to_centroid_to_prots = os.path.join(os.path.dirname(__file__), "centroid_to_prots")

    database = args.database.rstrip("/")

    # Load embeddings from intermediate folder
    embeddings_file = os.path.join(intermediate_folder, "query_embeddings.npy")
    print(f"Loading embeddings from {embeddings_file}")
    if not os.path.exists(embeddings_file):
        print(f"Error: Embeddings file not found at {embeddings_file}")
        exit(1)
    query_embeddings = np.load(embeddings_file)
    print(f"Loaded embeddings with shape {query_embeddings.shape}")
    
    query_embeddings_full = query_embeddings.copy()
    original_cutoff = cutoff

    # Load and apply PCA if it exists in the database folder
    pca, n_pca_components = load_pca_if_exists(database)
    if pca is not None:
        print(f"Found PCA model in database folder. Applying PCA with {n_pca_components} components...")
        query_embeddings = apply_pca_to_query(query_embeddings, pca)
        print(f"Query embeddings transformed to shape {query_embeddings.shape}")
        cutoff = cutoff - 0.03 # decrease cutoff to account for PCA dimensionality reduction, which shrinks angles
    else:
        print("No PCA model found. Using original 512-dimensional embeddings.")

    #check that the embeddings are normalized, if not normalize them and print a warning
    norms = np.linalg.norm(query_embeddings, axis=1)
    if not np.allclose(norms, 1, atol=1e-3):
        if pca is None:
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
        
    query_names_full = query_names.copy()

    print(f"Number of queries: {len(query_names)}")

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
            max_workers=args.index_threads
        )
        db_name = "usearch"
        time_taken = time.time() - t1
        print(f"Completed searching the vector database in {time_taken:.2f} seconds")
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
            max_workers=args.index_threads
        )
        db_name = "FAISS"
        
    # filter pca hits if PCA was active
    if pca is not None and os.path.exists(os.path.join(database, "centroids.fa.embeddings")):
        embeddings_file_path = os.path.join(database, "centroids.fa.embeddings")
        # filter PCA hits using the original cut-off if possible (ignoring max_dst offset or keeping it)
        # using the original base cutoff
        filter_cutoff = original_cutoff
        query_results = filter_pca_hits(query_embeddings_full, query_names_full, query_results, embeddings_file_path, filter_cutoff, max_workers=args.index_threads)
    else:
        # Strip the c_idx element since we didn't use filter_pca_hits
        query_results = [res[:4] for res in query_results]
    
    t2 = time.time()

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

    # Align centroids against query using MMseqs2 to filter results
    t_align_start = time.time()
    matched_centroids = align_centroids_with_mmseqs2(unique_fasta, args.query_sequences, args.align_threads, intermediate_folder)
    t_align_end = time.time()

    # Filter query_results to keep only centroids that aligned with the query
    # Extract centroid IDs from the matched set (handling '>' prefix and header parsing)
    matched_centroid_ids = set()
    for centroid_id in matched_centroids:
        # Remove '>' if present and get the first word
        clean_id = centroid_id.lstrip(">").split()[0]
        matched_centroid_ids.add(clean_id)

    filtered_query_results = []
    
    if args.deep_search:
        # Deep search: use all query results without alignment filtering
        print("Using deep-search mode: extracting proteins from ALL search results (no alignment filtering)")
        filtered_query_results = query_results
    else:
        # Normal mode: filter to only aligned centroids
        for query_name, centroid_name, sequence, distance in query_results:
            # Extract centroid ID from centroid_name for comparison
            centroid_id = centroid_name.strip()[1:].split()[0]  # Remove '>' and take first word
            if centroid_id in matched_centroid_ids:
                filtered_query_results.append((query_name, centroid_name, sequence, distance))

    print(f"Filtered results: {len(filtered_query_results)} results from {len(query_results)} (kept centroids alignable with query)" if not args.deep_search else f"Deep-search mode: processing all {len(filtered_query_results)} search results")

    # Output all query results as diversified_hits.tsv (main output file) - includes all centroids for user research
    diversified_hits_file = os.path.join(output_folder, "diversified_hits.tsv")
    with open(diversified_hits_file, "w") as tsvfile:
        tsvfile.write("#query_name\tresult_name\tresult_sequences\tcosine_distance\n")
        for query_name, centroid_name, sequence, distance in query_results:
            tsvfile.write(f"{query_name.strip()}\t{centroid_name.strip()[1:].split()[0]}\t{sequence}\t{distance}\n")
    print(f"Diversified hits TSV file written: {diversified_hits_file}")

    t3 = time.time()

    # Optionally continue with protein extraction and full alignment
    if len(filtered_query_results) > 0:
        centroid_hits = list(set([x[1] for x in filtered_query_results]))
        all_results = obtain_all_proteins(centroid_hits, database+"/all_prots", path_to_centroid_to_prots, args.index_threads)
        t4_start = time.time()

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
        mmseqs2_results(all_results, args.query_sequences, args.outfmt, mmseqs2_output, args.align_threads, intermediate_folder)
        t4 = time.time()

        # Create top hit file in intermediate_files
        top_hit_file = os.path.join(intermediate_folder, "matches.top_hit")
        command = f"awk '!seen[$1]++' {mmseqs2_output} > {top_hit_file}"  # to keep only the first hit
        subprocess.run(command, shell=True, check=True)
    else:
        t4 = t3
        print("No filtered results, skipping protein extraction and MMseqs2 alignment")

    print(f"Time for querying {db_name} database: {t2 - t1:.2f} seconds")
    print(f"Time for centroid alignment: {t_align_end - t_align_start:.2f} seconds")
    print(f"Time for obtaining all proteins: {t4_start - t3:.2f} seconds" if len(filtered_query_results) > 0 else "No filtered results, skipping protein extraction")
    print(f"Time for running MMseqs2: {t4 - t4_start:.2f} seconds" if len(filtered_query_results) > 0 else "No filtered results, skipping MMseqs2 alignment")
    print(f"Total time: {t4 - t1:.2f} seconds")
    
