#!/usr/bin/env python3
"""
Search a FAISS/usearch database using pre-computed query embeddings.
"""
import os

import faiss
import argparse
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import numpy as np
import sys
import tempfile
import subprocess
import mmap
import shutil
from sklearn.metrics.pairwise import cosine_distances
from usearch.index import Index,search, MetricKind, BatchMatches
import datetime
import gzip

__version__ = "3.1.1"

WORKER_QUERY_EMBEDDINGS = None


def _init_search_worker(query_embeddings):
    global WORKER_QUERY_EMBEDDINGS
    WORKER_QUERY_EMBEDDINGS = query_embeddings


def _cat_files(input_files, output_file):
    """Concatenate files on disk into a single output file.

    Uses batched cat calls to avoid hitting OS ARG_MAX with very large file lists.
    """
    batch_size = 1000
    with open(output_file, "wb") as out_f:
        if input_files:
            for i in range(0, len(input_files), batch_size):
                batch = input_files[i:i + batch_size]
                subprocess.run(["cat", *batch], check=True, stdout=out_f)
        else:
            out_f.truncate(0)


def _sort_tsv_on_disk(input_file, output_file, key_specs, parallel_threads=1):
    """Sort a TSV file on disk using GNU sort."""
    sort_cmd = ["sort", "-T", os.path.dirname(output_file) or "."]
    if parallel_threads and parallel_threads > 1:
        sort_cmd.extend(["--parallel", str(parallel_threads)])
    sort_cmd.extend(["-t", "\t"])
    for field_number, modifier in key_specs:
        sort_cmd.append(f"-k{field_number},{field_number}{modifier}")
    sort_cmd.extend([input_file, "-o", output_file])
    subprocess.run(sort_cmd, check=True)


def _write_results_fasta_from_tsv(tsv_file, fasta_file):
    """Stream a TSV file and write the corresponding FASTA file."""
    with open(tsv_file, "r") as in_f, open(fasta_file, "w") as out_f:
        for line in in_f:
            if not line or line.startswith("#"):
                continue
            query_name, centroid_name, sequence, _distance = line.rstrip("\n").split("\t", 3)
            out_f.write(f">{centroid_name}#{query_name}\n{sequence}\n")


def _write_unique_centroids_from_tsv(tsv_file, fasta_file):
    """Create a deduplicated centroid FASTA from a TSV file sorted by centroid columns."""
    with open(tsv_file, "r") as in_f, open(fasta_file, "w") as out_f:
        last_pair = None
        for line in in_f:
            if not line or line.startswith("#"):
                continue
            _query_name, centroid_name, sequence, _distance = line.rstrip("\n").split("\t", 3)
            pair = (centroid_name, sequence)
            if pair == last_pair:
                continue
            last_pair = pair
            out_f.write(f">{centroid_name}\n{sequence}\n")


def _load_filtered_results_from_tsv(tsv_file, matched_centroid_ids):
    """Load only rows whose centroid is in matched_centroid_ids."""
    filtered_query_results = []
    with open(tsv_file, "r") as in_f:
        for line in in_f:
            if not line or line.startswith("#"):
                continue
            query_name, centroid_name, sequence, distance_str = line.rstrip("\n").split("\t", 3)
            centroid_id = centroid_name.strip().split()[0]
            if centroid_id in matched_centroid_ids:
                filtered_query_results.append((query_name, centroid_name, sequence, float(distance_str)))
    return filtered_query_results


def _load_all_results_from_tsv(tsv_file):
    """Load every row from a named TSV into RAM."""
    all_results = []
    with open(tsv_file, "r") as in_f:
        for line in in_f:
            if not line.strip():
                continue
            query_name, centroid_name, sequence, distance_str = line.rstrip("\n").split("\t", 3)
            all_results.append((query_name, centroid_name, sequence, float(distance_str)))
    return all_results

def query_bin(bin_file, original_fasta, database_folder, subdatabase_size, cutoff, tmp_results_dir):
    start_time = time.time()
    faiss.omp_set_num_threads(1)

    file_starting_pos = int(bin_file.strip(".bin").split("_")[2])*subdatabase_size
    index_path = os.path.join(database_folder, bin_file)
    index = faiss.read_index(index_path)
    nb_of_searches = 1
    distances = []
    query_embeddings = WORKER_QUERY_EMBEDDINGS
    while len(distances) == 0 or (distances[0][-1] < cutoff and nb_of_searches <= 10):
        k = 20 * 2 ** (nb_of_searches)
        distances, indices = index.search(query_embeddings, k=k)
        nb_of_searches += 1

    hits_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=tmp_results_dir,
        prefix=f"{bin_file}.",
        suffix=".hits.tsv",
    )
    distinct_matches = set()
    for query_idx, (dist_row, idx_row) in enumerate(zip(distances, indices)):
        for dist, idx in zip(dist_row, idx_row):
            if dist < cutoff:
                match_idx = int(file_starting_pos + idx)
                distinct_matches.add(match_idx)
                hits_file.write(f"{match_idx}\t{query_idx}\t{dist}\n")

    hits_file.close()

    print(f"Total distinct matches for bin {bin_file}: {len(distinct_matches)}")
    
    elapsed_time = time.time() - start_time
    print(f"Time taken for querying bin {bin_file}: {elapsed_time:.2f} seconds")
    return hits_file.name

def search_faiss_database(original_fasta, database_folder, query_embeddings, query_names, tmp_folder, cutoff=0.2, subdatabase_size=10000000, max_workers=4):
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

    faiss_database_folder = database_folder+"/faiss"
    bin_files = [file for file in os.listdir(faiss_database_folder) if file.endswith(".bin")]

    tmp_results_dir = tempfile.mkdtemp(prefix="faiss_hits_", dir=tmp_folder)
    bin_result_files = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_search_worker, initargs=(query_embeddings,)) as executor:
        futures = [
            executor.submit(query_bin, bin_file, original_fasta, faiss_database_folder, subdatabase_size, cutoff, tmp_results_dir) for bin_file in bin_files
        ]
        for future in as_completed(futures):
            bin_result_files.append(future.result())

    merged_hits_file = os.path.join(tmp_results_dir, "all_hits.tsv")
    _cat_files(bin_result_files, merged_hits_file)
    for path in bin_result_files:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    return merged_hits_file


def query_usearch_bin(bin_file, original_fasta, database_folder, subdatabase_size, cutoff, tmp_results_dir):
    """Query a single usearch index file and return results as dict: match_idx -> [(query_idx, distance), ...]"""
    start_time = time.time()

    file_starting_pos = int(bin_file.strip(".bin").split("_")[2]) * subdatabase_size
    index_path = os.path.join(database_folder, bin_file)
    
    # Load the usearch index
    index = Index(ndim=512, metric='cos')
    index.load(index_path)
    
    # Search with increasing k until we get results beyond the cutoff
    nb_of_searches = 1
    all_matches = None
    query_embeddings = WORKER_QUERY_EMBEDDINGS
    while nb_of_searches <= 10:
        k = 2000 * 2 ** nb_of_searches
        matches : BatchMatches = index.search(query_embeddings, k, exact=False)
        if query_embeddings.shape[0] == 1:
            matches = [matches]

        # Check if we got far enough results
        max_distance = 0
        for i in range(len(matches)):
            if len(matches[i]) > 0:
                last_valid_idx = len(matches[i]) - 1
                max_distance = max(max_distance, matches[i][last_valid_idx].distance)
        
        if max_distance >= cutoff or nb_of_searches >= 10:
            all_matches = matches
            break
        
        nb_of_searches += 1
    
    if all_matches is None:
        return {}

    hits_file = tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=tmp_results_dir,
        prefix=f"{bin_file}.",
        suffix=".hits.tsv",
    )
    distinct_matches = set()
    for query_idx in range(len(all_matches)):
        for j in range(len(all_matches[query_idx])):
            dist = all_matches[query_idx][j].distance
            if dist < cutoff:
                key = all_matches[query_idx][j].key
                match_idx = int(file_starting_pos + key)
                distinct_matches.add(match_idx)
                hits_file.write(f"{match_idx}\t{query_idx}\t{dist}\n")

    hits_file.close()

    print(f"Total distinct matches for bin {bin_file}: {len(distinct_matches)}")
    
    elapsed_time = time.time() - start_time
    print(f"Time taken for querying bin {bin_file}: {elapsed_time:.2f} seconds")
    return hits_file.name

def search_usearch_database(original_fasta, database_folder, query_embeddings, query_names, tmp_folder, cutoff=0.2, subdatabase_size=10000000, max_workers=4):
    """
    Searches a usearch database with pre-computed embeddings and retrieves the nearest neighbors in parallel.

    Args:
        original_fasta (str): Path to the original FASTA file.
        database_folder (str): Path to the folder containing usearch database files (.bin).
        query_embeddings (np.ndarray): Pre-computed embeddings for query sequences.
        query_names (list of str): List of query names.
        tmp_folder (str): Path to the temporary folder for storing intermediate results.
        cutoff (float): Distance cutoff for results.
        subdatabase_size (int): Number of vectors in each usearch subdatabase.
        max_workers (int): Number of parallel workers.

    Returns:
        list of tuple: Each tuple contains (query_name, name_line, sequence_line, distance).
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    usearch_database_folder = database_folder + "/usearch"
    bin_files = [file for file in os.listdir(usearch_database_folder) if file.endswith(".bin")]
    
    # # #DEBUG: Only use one random bin file to speed up testing
    # bin_file_random = np.random.randint(0,len(bin_files))
    # bin_files = bin_files[bin_file_random:bin_file_random+100]
    # print("DEBUUUUUGUGkkk")

    tmp_results_dir = tempfile.mkdtemp(prefix="usearch_hits_", dir=tmp_folder)
    bin_result_files = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_search_worker, initargs=(query_embeddings,)) as executor:
        futures = [
            executor.submit(query_usearch_bin, bin_file, original_fasta, usearch_database_folder, subdatabase_size, cutoff, tmp_results_dir)
            for bin_file in bin_files
        ]
        for future in as_completed(futures):
            bin_result_files.append(future.result())

    time1 = time.time()
    merged_hits_file = os.path.join(tmp_results_dir, "all_hits.tsv")
    _cat_files(bin_result_files, merged_hits_file)
    for path in bin_result_files:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    print(f"Time for concatenating all results : {time.time() - time1:.2f} seconds")
    return merged_hits_file


def _read_embedding_batch(args):
    """Helper function for parallel embedding reading (full 512 dims). Must be at module level for ThreadPoolExecutor.
    Returns embeddings as a list in the same order as input indices (NOT normalized).
    Uses buffered I/O to reduce seek operations: loads 100 embeddings at a time.
    """
    embeddings_file_path, match_indices_batch = args
    BUFFER_SIZE = 100 # Load 100 embeddings per seek
    EMBEDDING_SIZE_BYTES = 512 * 2
    
    # Use dict to maintain index->embedding mapping, then return as list in original order
    result_list = [None] * len(match_indices_batch)  # Placeholder for results in original order
    buffer_bytes = b''  # Cache of loaded embeddings
    buffer_start = None
    buffer_end = None
    nb_seeks = 0
    
    # Aggregate timing stats
    nb_cache_hits = 0
    nb_cache_misses = 0
    total_seek_time = 0.0
    total_read_time = 0.0
    total_frombuffer_hit_time = 0.0
    total_frombuffer_miss_time = 0.0
    
    with open(embeddings_file_path, "rb") as embeddings_file:
        for match_idx_idx, match_idx in enumerate(match_indices_batch):
            # Check if idx is in current buffer
            if buffer_start is not None and buffer_start <= match_idx < buffer_end:
                # Cache hit: time the frombuffer operation
                # t_frombuffer_start = time.time()
                result_list[match_idx_idx] = np.frombuffer(buffer_bytes[(match_idx - buffer_start) * EMBEDDING_SIZE_BYTES:(match_idx - buffer_start + 1) * EMBEDDING_SIZE_BYTES], dtype=np.float16).astype(np.float32)
                # total_frombuffer_hit_time += time.time() - t_frombuffer_start
                nb_cache_hits += 1
            else:
                # Cache miss: time seek, read, and frombuffer separately
                nb_seeks += 1
                nb_cache_misses += 1
                buffer_start = match_idx
                buffer_end = match_idx + BUFFER_SIZE
                
                # t_seek_start = time.time()
                embeddings_file.seek(EMBEDDING_SIZE_BYTES * match_idx)
                # total_seek_time += time.time() - t_seek_start
                
                # t_read_start = time.time()
                buffer_bytes = embeddings_file.read(EMBEDDING_SIZE_BYTES * BUFFER_SIZE)
                # total_read_time += time.time() - t_read_start
                
                # t_frombuffer_start = time.time()
                result_list[match_idx_idx] = np.frombuffer(buffer_bytes[:EMBEDDING_SIZE_BYTES], dtype=np.float16).astype(np.float32)
                # total_frombuffer_miss_time += time.time() - t_frombuffer_start

    # Print aggregated timing summary
    total_time = total_seek_time + total_read_time + total_frombuffer_hit_time + total_frombuffer_miss_time
    # print(f"Timing summary for _read_embedding_batch ({len(match_indices_batch)} embeddings):")
    # print(f"  Cache hits: {nb_cache_hits}, Cache misses: {nb_cache_misses} ({nb_seeks} seeks)")
    # print(f"  Seek time (total): {total_seek_time*1000:.3f}ms (avg per miss: {total_seek_time*1000/max(1,nb_cache_misses):.3f}ms)")
    # print(f"  Read time (total): {total_read_time*1000:.3f}ms (avg per miss: {total_read_time*1000/max(1,nb_cache_misses):.3f}ms)")
    # print(f"  Frombuffer time (hits): {total_frombuffer_hit_time*1000:.3f}ms (avg: {total_frombuffer_hit_time*1000/max(1,nb_cache_hits):.3f}ms)")
    # print(f"  Frombuffer time (misses): {total_frombuffer_miss_time*1000:.3f}ms (avg: {total_frombuffer_miss_time*1000/max(1,nb_cache_misses):.3f}ms)")
    # print(f"  Total time: {total_time*1000:.3f}ms")
    return result_list


def _read_fasta_batch(args):
    """Helper function for parallel FASTA reading. Must be at module level for ThreadPoolExecutor."""
    original_fasta, match_indices_batch, index_positions = args
    batch_data = {}
    
    with open(original_fasta, "rb") as fastafile:
        for match_idx in match_indices_batch:
            if match_idx not in index_positions:
                continue
            pos = index_positions[match_idx]
            fastafile.seek(pos)
            
            # Read header and sequence
            name_line = fastafile.readline().rstrip(b'\n').decode('utf-8', errors='ignore').strip()
            sequence_line = fastafile.readline().rstrip(b'\n').decode('utf-8', errors='ignore').strip()
            
            batch_data[match_idx] = (name_line, sequence_line)
    
    return batch_data


def load_names_from_results(query_results, query_names, original_fasta, database_folder, parallel_threads=4):
    """
    Load FASTA names and sequences from results dict using sorted key access for I/O efficiency.
    
    Optimization: Use mmap for .names file (instant, OS pages on demand), binary mode + parallel I/O for FASTA.
    
    Args:
        query_results: dict mapping match_idx to list of (query_idx, distance) tuples (already sorted by match_idx)
        query_names: list of query names indexed by query_idx
        original_fasta: path to FASTA file
        database_folder: path to folder containing .names file
        parallel_threads: number of threads for parallel FASTA I/O (default 4, use 1 to disable)
    
    Returns:
        list of (query_name, centroid_name, sequence, distance) tuples
    """
    # parallel_threads = min(parallel_threads,4) #use at most 4 threads for reading FASTA, more threads won't help due to disk I/O limits and GIL 
    if len(query_results) == 0:
        return []
    
    # Count total results to estimate time
    total_results = sum(len(v) for v in query_results.values())
    time_start = time.time()
    
    names_file = os.path.join(database_folder, f"{os.path.basename(original_fasta)}.names")
    
    # Step 1: Use mmap for .names file (instant, OS pages on demand)
    mmap_start = time.time()
    index_positions = {}
    
    with open(names_file, "rb") as nf:
        with mmap.mmap(nf.fileno(), 0, access=mmap.ACCESS_READ) as names_mmap:
            for match_idx in query_results.keys():
                offset = 8 * match_idx
                if offset + 8 <= len(names_mmap):
                    position = int.from_bytes(names_mmap[offset:offset+8], byteorder='little', signed=False)
                    index_positions[match_idx] = position
                else:
                    print("ERROR 2212: the subdatabas size is likely wrongly hardcoded" , match_idx, offset, len(names_mmap))
                    sys.exit(1)
    
    # print(f"Mapped {len(index_positions)} positions in {time.time() - mmap_start:.2f}s")
    
    # Step 2: Read FASTA in parallel (ThreadPoolExecutor for shared file descriptors)
    # print(f"Reading FASTA sequences using {parallel_threads} parallel threads...")
    fasta_read_start = time.time()
    
    index_to_data = {}
    match_indices_list = list(query_results.keys())
    
    # Split indices into batches for parallel processing
    batch_size = max(1, len(match_indices_list) // parallel_threads)
    batches = [match_indices_list[i:i+batch_size] for i in range(0, len(match_indices_list), batch_size)]
    
    # Use ThreadPoolExecutor for I/O parallelization (threads share file descriptors efficiently)
    with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        futures = [
            executor.submit(_read_fasta_batch, (original_fasta, batch, index_positions))
            for batch in batches
        ]
        for future in as_completed(futures):
            batch_data = future.result()
            index_to_data.update(batch_data)
    
    # print(f"Read FASTA sequences in {time.time() - fasta_read_start:.2f}s")
    
    # Build final results list (iterate in sorted order)
    final_results = []
    results_loaded = 0
    last_progress_count = 0
    
    for match_idx in query_results.keys():
        if match_idx not in index_to_data:
            continue
        name_line, sequence_line = index_to_data[match_idx]
        for query_idx, distance in query_results[match_idx]:
            final_results.append((query_names[query_idx], name_line, sequence_line, distance))
            results_loaded += 1
            
    # print(f"Total time to load names: {time.time() - time_start:.2f} seconds")
    
    return final_results


def load_names_from_hits_file(hits_file, query_names, original_fasta, database_folder, parallel_threads=4, output_tsv=None):
    """Load FASTA names and sequences from a sorted hits TSV and write a named TSV on disk."""
    if output_tsv is None:
        output_tsv = os.path.join(os.path.dirname(hits_file), "query_results_named.tsv")

    if len(query_names) == 0:
        with open(output_tsv, "w") as out_f:
            pass
        return output_tsv

    names_file = os.path.join(database_folder, f"{os.path.basename(original_fasta)}.names")

    unique_match_indices = []
    last_match_idx = None
    with open(hits_file, "r") as in_f:
        for line in in_f:
            if not line.strip():
                continue
            match_idx = int(line.split("\t", 1)[0])
            if match_idx != last_match_idx:
                unique_match_indices.append(match_idx)
                last_match_idx = match_idx

    index_positions = {}
    with open(names_file, "rb") as nf:
        with mmap.mmap(nf.fileno(), 0, access=mmap.ACCESS_READ) as names_mmap:
            for match_idx in unique_match_indices:
                offset = 8 * match_idx
                if offset + 8 <= len(names_mmap):
                    position = int.from_bytes(names_mmap[offset:offset+8], byteorder='little', signed=False)
                    index_positions[match_idx] = position
                else:
                    print("ERROR 2212: the subdatabas size is likely wrongly hardcoded", match_idx, offset, len(names_mmap))
                    sys.exit(1)

    index_to_data = {}
    match_indices_list = unique_match_indices
    batch_size = max(1, len(match_indices_list) // max(1, parallel_threads))
    batches = [match_indices_list[i:i+batch_size] for i in range(0, len(match_indices_list), batch_size)]

    with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
        futures = [executor.submit(_read_fasta_batch, (original_fasta, batch, index_positions)) for batch in batches]
        for future in as_completed(futures):
            batch_data = future.result()
            index_to_data.update(batch_data)

    with open(hits_file, "r") as in_f, open(output_tsv, "w") as out_f:
        for line in in_f:
            if not line.strip():
                continue
            match_idx_str, query_idx_str, distance_str = line.rstrip("\n").split("\t", 2)
            match_idx = int(match_idx_str)
            query_idx = int(query_idx_str)
            if match_idx not in index_to_data:
                continue
            name_line, sequence_line = index_to_data[match_idx]
            out_f.write(f"{query_names[query_idx]}\t{name_line.lstrip('>')}\t{sequence_line}\t{distance_str}\n")

    return output_tsv


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


def _format_protein_fasta_header(name):
    if '>' in name:
        name = name.lstrip('>')
    if len(name.split()[0].split("_")) < 4:  # that's because the human and nonhuman db are not exactly formatted the same way
        header_parts = ''.join(name.split()[1:])
        accession = name.split()[0][1:]
        return f">{header_parts}#{accession}"

    header_parts = "_".join(name.split("_")[3:])
    header_clean = "".join(header_parts.split())
    accession = "_".join(name.split("_")[:3])
    return f">{header_clean}#{accession}"


def obtain_all_proteins(centroids, database_all_proteins, path_to_centroid_to_prots, num_threads, output_file):
    """
    For each result, runs an external command to extract proteins and writes them directly to a FASTA file.
    """
    # unique_results = set()
    # # Use ProcessPoolExecutor instead of ThreadPoolExecutor to avoid GIL contention
    # with open(output_file, "w") as out_f:
    #     with ProcessPoolExecutor(max_workers=num_threads) as executor:
    #         # Pass all arguments as a tuple since process_result needs to be at module level
    #         futures = [executor.submit(_process_centroid_result, (centroid, database_all_proteins, path_to_centroid_to_prots)) for centroid in centroids]
    #         for i, future in enumerate(as_completed(futures), 1):
    #             proteins = future.result()
    #             for name, seq in proteins:
    #                 fasta_header = _format_protein_fasta_header(name)
    #                 fasta_entry = (fasta_header, seq)
    #                 if fasta_entry in unique_results:
    #                     continue
    #                 unique_results.add(fasta_entry)
    #                 out_f.write(f"{fasta_header}\n{seq}\n")

    #Actually, do a different strategy with the single files
    protein_file = database_all_proteins + "proteins.fasta.zst"
    protein_index_file = database_all_proteins + "proteins.fasta.zst.index"
    #index file is a TSV with three columns: protein_id (sorted), frame position in the zst file and length of frame in bytes. We can use this index to extract the proteins we want without decompressing the whole file

    #start by sorting the centroids
    sorted_centroids = sorted(centroids)

    #now go through the index file and extract the proteins we want
    with open(protein_index_file, "r") as index_f, open(protein_file, "r") as protein_f, open(output_file, "w") as out_f:
        next_protein_id_to_look_at = 0
        next_protein_name = sorted_centroids[next_protein_id_to_look_at]
        for enumerate(l,line) in index_f:
            protein_id, frame_position, frame_length = line.strip().split("\t")
            if protein_id == next_protein_name:
                # Extract the protein from the zst file
                protein_f.seek(int(frame_position))
                protein_data = protein_f.read(int(frame_length))
                out_f.write(protein_data)
            
                next_protein_id_to_look_at += 1
                next_protein_name = sorted_centroids[next_protein_id_to_look_at]
                if next_protein_id_to_look_at >= len(sorted_centroids):
                    break
            
            if l % 10000 == 0:
                print(f"Processed {l} lines in the index file. Found {next_protein_id_to_look_at} proteins so far.")

    print("Finished fishing out the proteins from the zst file. Now decompressing the output to a fasta file...")
    sys.exit(0)

    #the output is a zst file, so we need to decompress it to a fasta file
    os.system(f"zstd -d {output_file} -o {output_file.tmp} && mv {output_file.tmp} {output_file}")


def mmseqs2_results(original_query_fasta, returned_sequences_fasta, output_format, output_file, output_fasta_file, num_threads, intermediate_folder):
    """
    Run MMseqs2 search between two existing FASTA files.
    """

    print("Running MMseqs2 now...")

    original_query_mmseqs = os.path.join(intermediate_folder, "original_query_mmseqs")
    returned_sequences_mmseqs = os.path.join(intermediate_folder, "returned_sequences_mmseqs")
    result_mmseqs = os.path.join(intermediate_folder, "result_mmseqs")
    tmp_mmseqs = os.path.join(intermediate_folder, "tmp_mmseqs")

    # Set up log files for mmseqs2 output
    log_file_1 = os.path.join(intermediate_folder, "mmseqs_createdb_1.log") if intermediate_folder else os.devnull
    log_file_2 = os.path.join(intermediate_folder, "mmseqs_createdb_2.log") if intermediate_folder else os.devnull
    log_file_search = os.path.join(intermediate_folder, "mmseqs_search.log") if intermediate_folder else os.devnull
    log_file_convertalis = os.path.join(intermediate_folder, "mmseqs_convertalis.log") if intermediate_folder else os.devnull

    with open(log_file_1, "w") as lf1, open(log_file_2, "w") as lf2, open(log_file_search, "w") as lfs, open(log_file_convertalis, "w") as lfc:
        subprocess.run(["mmseqs", "createdb", original_query_fasta, original_query_mmseqs], check=True, stdout=lf1, stderr=subprocess.STDOUT)
        subprocess.run(["mmseqs", "createdb", returned_sequences_fasta, returned_sequences_mmseqs], check=True, stdout=lf2, stderr=subprocess.STDOUT)

    with open(log_file_search, "w") as lfs:
        # print(f"Running command: mmseqs search {returned_sequences_mmseqs} {original_query_mmseqs} {result_mmseqs} {tmp_mmseqs} --threads {num_threads}")
        #remove the result database if it already exists to avoid errors
        os.system(f"rm -rf {result_mmseqs}*")
        subprocess.run([
            "mmseqs", "search", returned_sequences_mmseqs, original_query_mmseqs, result_mmseqs, tmp_mmseqs,
            "--threads", str(num_threads)
        ], check=True, stdout=lfs, stderr=subprocess.STDOUT)


    print(f"Output format for MMseqs2 convertalis: {output_format}")
    if output_format == "0":
        with open(output_file, "w") as out_f:
            out_f.write("#target\tquery\tidentity\talignment_length\tnb_mismatches\tnb_gap_openings\ttarget_start\ttarget_end\tquery_start\tquery_end\tevalue\tbitscore\n")

    with open(log_file_convertalis, "w") as lfc:
        # print(f"Running command: mmseqs convertalis {original_query_mmseqs} {returned_sequences_mmseqs} {result_mmseqs} {output_file} --format-mode {output_format}")
        subprocess.run([
            "mmseqs", "convertalis", returned_sequences_mmseqs, original_query_mmseqs, result_mmseqs, output_file,
            "--format-mode", output_format
        ], check=True, stdout=lfc, stderr=subprocess.STDOUT)

    #now create a FASTA file with the aligned sequences: load the name of all aligned sequences in the mmseqs2 output and extract them from the target FASTA file
    aligned_ids = set()
    with open(output_file, "r") as resf:
        for line in resf:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) > 0:
                aligned_ids.add(fields[0])  # First column is the target ID

    print(f"Found {len(aligned_ids)} aligned sequences in MMseqs2 results: {list(aligned_ids)[0:5]}... Now looking for them in {returned_sequences_fasta} to create {output_fasta_file}")

    with open(returned_sequences_fasta, "r") as target_f, open(output_fasta_file, "w") as out_f:
        name = None
        seq = []
        for line in target_f:
            line = line.strip()
            if line.startswith(">"):
                if name is not None and name in aligned_ids:
                    out_f.write(f"{name}\n{''.join(seq)}\n")
                seq = []
            else:
                seq.append(line)
        if name is not None and name in aligned_ids:
            out_f.write(f"{name}\n{''.join(seq)}\n")


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
        result_tsv = os.path.join(intermediate_folder, "result.tsv")
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
                    matched_centroids.add(centroid_id.lstrip(">").split()[0])  # Store only the first part of the header

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
    
    # RAM needed per thread is empirically: bin_size * 3
    ram_per_thread_gb = bin_size_gb * 6
    
    if ram_per_thread_gb <= 0:
        print("Warning: Bin file is empty or too small. Defaulting to 1 thread.")
        return 1
    
    # Calculate how many threads we can run with available memory
    threads_by_memory = int(max_memory_gb / ram_per_thread_gb)
    
    # Use the minimum of threads limited by memory or max_threads
    index_threads = max(1, min(threads_by_memory, max_threads))
    
    print(f"Bin file size: {bin_size_gb:.2f} GB, estimated RAM per thread: {ram_per_thread_gb:.2f} GB")
    print(f"Using {index_threads} threads for index querying based on the available memory and bin file size.")
    
    return index_threads

def check_input_fasta(input_fasta):
    """
    Check if the input FASTA file exists, is readable, and is proteins (not nucleotides). Raises an error if any check fails.
    
    Args:
        input_fasta (str): Path to the input FASTA file.
    """

    with open(input_fasta, "r") as f:
        first_line = f.readline().strip()
        if not first_line.startswith(">"):
            raise ValueError(f"Input FASTA file {input_fasta} does not appear to be a valid FASTA file (missing header line).")
        
        # Read the first sequence line
        second_line = f.readline().strip()
        if not second_line:
            raise ValueError(f"Input FASTA file {input_fasta} does not contain any sequences.")
        
        # Check for nucleotide characters (A, C, G, T, N)
        nucleotide_chars = set("ACGTNacgtn")
        sequence_chars = set(second_line)
        if sequence_chars.issubset(nucleotide_chars):
            raise ValueError(f"Input FASTA file {input_fasta} appears to contain nucleotide sequences. Please provide protein sequences.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search a database (FAISS or Usearch) using pre-computed embeddings. Can optionally embed query sequences on-the-fly.")
    parser.add_argument("--query_sequences", required=True, help="Fasta file of queries")
    parser.add_argument("--database", required=True, help="Path to the folder containing database files.")
    parser.add_argument("--output", "-o", required=True, help="Path to the output folder (created by embed_query.py if embedding step was run separately)")
    parser.add_argument("--db-type", type=str, choices=['faiss', 'usearch'], default='usearch', help="Database type to use: faiss or usearch (default: usearch)")
    parser.add_argument("--outfmt", type=str, default='0', help="Format of the mmseqs2 output [0], default is 0 which is a tabular format with header. See mmseqs2 documentation for details.")
    parser.add_argument("-m", "--memory", type=float, required=True, help="Maximum memory available in GB (mandatory)")
    parser.add_argument("-t", "--num_threads", type=int, required=True, help="Maximum number of threads available (mandatory)")
    parser.add_argument("--force_cpu", action="store_true", help="Force the use of CPU even if GPUs are available (for embedding step).")
    # parser.add_argument("-r","--do_not_reduce_query", action="store_true", help="Do not cluster similar proteins to reduce time (identity > 0.9)")
    #parser.add_argument("--subdatabases_size", type=int, default=10000000, help="Number of vectors in each faiss database")
    #parser.add_argument("--cutoff", type=float, default=0.2, help="Distance cutoff for results")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Validate that query_sequences is provided for MMseqs2 alignment
    if not args.query_sequences:
        print("Error: --query_sequences is required for MMseqs2 alignment step.")
        print("Please provide the path to your query FASTA file.")
        exit(1)


    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("search_database.py version ", __version__)
    print("command line used:\n", " ".join(sys.argv))

    check_input_fasta(args.query_sequences)
    
    # Set up folder structure
    output_folder = args.output.rstrip("/")
    intermediate_folder = os.path.join(output_folder, "intermediate_files")
    embeddings_file = os.path.join(intermediate_folder, "query_embeddings.npy")

    # reduce_query = not args.do_not_reduce_query
    reduce_query = False
    
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
        
        # Create intermediate_files folder
        os.makedirs(intermediate_folder, exist_ok=True)
        
        # Embed the sequences
        batch_size = 10
        embed_query_sequences(
            query_file=args.query_sequences,
            gpus_available=gpus_available,
            batch_size=batch_size,
            reduce_query=reduce_query,
            intermediate_folder=intermediate_folder
        )
        
    else:
        print(f"Found existing embeddings in {output_folder}")
    
    
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
    
    cutoff = 0.2 #cosine distance cutoff
    subdatabase_size = 100_000 #this is linked to the size of the databases, do not change this unless you know what you are doing
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
        
    query_names_full = query_names.copy()

    print(f"Number of queries: {len(query_names)}")

    t1 = time.time()
    
    # Choose database type
    if args.db_type == 'usearch':
        raw_hits_file = search_usearch_database(
            original_fasta=database+"/centroids.fa",
            database_folder=database,
            query_embeddings=query_embeddings,
            query_names=query_names,
            tmp_folder=intermediate_folder,
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
        raw_hits_file = search_faiss_database(
            original_fasta=database+"/centroids.fa",
            database_folder=database,
            query_embeddings=query_embeddings,
            query_names=query_names,
            cutoff=cutoff,
            subdatabase_size=subdatabase_size,
            tmp_folder=intermediate_folder,
            max_workers=args.index_threads
        )
        db_name = "FAISS"

    # Sort hits on disk by match index, then stream names and sequences into a TSV.
    sorted_hits_file = os.path.join(intermediate_folder, "query_results_hits_sorted.tsv")
    _sort_tsv_on_disk(raw_hits_file, sorted_hits_file, [(1, "n")], parallel_threads=args.index_threads)

    named_results_tsv = os.path.join(intermediate_folder, "query_results_named.tsv")
    load_names_from_hits_file(
        sorted_hits_file,
        query_names_full,
        database + "/centroids.fa",
        database,
        parallel_threads=args.align_threads,
        output_tsv=named_results_tsv,
    )

    # Sort the named results on disk by query name and distance
    sorted_named_results_tsv = os.path.join(intermediate_folder, "query_results_sorted.tsv")
    _sort_tsv_on_disk(named_results_tsv, sorted_named_results_tsv, [(1, ""), (4, "g")], parallel_threads=args.align_threads)

    total_named_results = 0
    with open(sorted_named_results_tsv, "r") as count_f:
        for _ in count_f:
            total_named_results += 1
    
    t2 = time.time()

    if total_named_results == 0:
        print("No results, exiting")
        print(f"Total time: {t2 - t1:.2f} seconds")
        sys.exit(0)

    # Output query_results as an intermediate FASTA file in intermediate_files
    intermediate_fasta = os.path.join(intermediate_folder, "query_results_intermediate.fasta")
    _write_results_fasta_from_tsv(sorted_named_results_tsv, intermediate_fasta)
    # print(f"Intermediate FASTA file written: {intermediate_fasta}")

    # Write a deduplicated intermediate FASTA file in intermediate_files (unique centroid name/seq pairs)
    unique_fasta = os.path.join(intermediate_folder, "unique_centroids.fasta")
    unique_sorted_tsv = os.path.join(intermediate_folder, "unique_centroids_sorted.tsv")
    _sort_tsv_on_disk(named_results_tsv, unique_sorted_tsv, [(2, ""), (3, "")], parallel_threads=args.align_threads)
    _write_unique_centroids_from_tsv(unique_sorted_tsv, unique_fasta)
    print(f"Unique centroid FASTA file written: {unique_fasta}")

    # Align centroids against query using MMseqs2 to filter results
    t_align_start = time.time()
    matched_centroids = align_centroids_with_mmseqs2(unique_fasta, args.query_sequences, args.align_threads, intermediate_folder)
    t_align_end = time.time()
    

    # Output all query results as PLM_similar_proteins.tsv (arguably main output file) - includes all centroids for user research
    diversified_hits_file = os.path.join(output_folder, "PLM_similar_proteins.tsv")
    with open(diversified_hits_file, "w") as out_f:
        out_f.write("#query_name\tresult_name\tresult_sequences\tcosine_distance\n")
        with open(sorted_named_results_tsv, "r") as in_f:
            shutil.copyfileobj(in_f, out_f)
    print(f"Diversified hits TSV file written: {diversified_hits_file}")

    # for temp_path in [raw_hits_file, sorted_hits_file, named_results_tsv, sorted_named_results_tsv, unique_sorted_tsv]:
    #     try:
    #         os.remove(temp_path)
    #     except FileNotFoundError:
    #         pass

    t3 = time.time()

    plm_query_results = _load_all_results_from_tsv(sorted_named_results_tsv)

    print(f"Total number of centroids hitting the query: {len(plm_query_results)}")

    # Optionally continue with protein extraction and full alignment
    if len(plm_query_results) > 0:
        centroid_hits = list(set([x[1] for x in plm_query_results]))
        fasta_output = os.path.join(output_folder, "all_proteins.fasta")
        obtain_all_proteins(centroid_hits, database+"/all_prots", path_to_centroid_to_prots, args.index_threads, fasta_output)
        #gzip the fasta file to save space
        time_now = time.time()
        with open(fasta_output, 'rb') as f_in:
            with gzip.open(fasta_output + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(fasta_output)
        print(f"All proteins FASTA file written: {fasta_output}.gz, compressed in {time.time() - time_now:.2f} seconds")
        t4_start = time.time()

        filtered_fasta_output = os.path.join(intermediate_folder, "all_proteins_filtered.fasta")
        filtered_query_results = _load_filtered_results_from_tsv(sorted_named_results_tsv, matched_centroids)
        centroids_fitered = list(set([x[1] for x in filtered_query_results]))
        obtain_all_proteins(centroids_fitered, database+"/all_prots", path_to_centroid_to_prots, args.index_threads, filtered_fasta_output)
        fasta_output = filtered_fasta_output

        # Run MMseqs2 and write main output files to output folder root
        mmseqs2_output = os.path.join(output_folder, "aligned_proteins.mmseqs2")
        fasta_mmseqs_output = os.path.join(output_folder, "aligned_proteins.fasta")
        mmseqs2_results(args.query_sequences, filtered_fasta_output, args.outfmt, mmseqs2_output, fasta_mmseqs_output, args.align_threads, intermediate_folder)
        t4 = time.time()

    print(f"Time for querying {db_name} database: {t2 - t1:.2f} seconds")
    print(f"Time for centroid alignment: {t_align_end - t_align_start:.2f} seconds")
    print(f"Time for obtaining all proteins: {t4_start - t3:.2f} seconds" if len(filtered_query_results) > 0 else "No filtered results, skipping protein extraction")
    print(f"Time for running MMseqs2: {t4 - t4_start:.2f} seconds" if len(filtered_query_results) > 0 else "No filtered results, skipping MMseqs2 alignment")
    print(f"Total time: {t4 - t1:.2f} seconds")
    
