import lzma
from Bio import SeqIO
import os
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import struct
import glob
import zstandard as zstd
from collections import defaultdict
from threading import Lock
import tempfile
import sys
import time
import subprocess
import datetime
import io
import pickle
from concurrent.futures import ThreadPoolExecutor

if hash('1') != -6863858015672933803:
    print("ERROR: do not forget to set the seed to 0:  PYTHONHASHSEED=0 python create_all_prots_database.py")
    print(hash('1'))
    sys.exit(1)

if len(sys.argv) != 1:
    print("Usage: python create_all_prots_database.py")
    sys.exit(1)

file_locks = {}

# num_shards= 100
# num_centroids_shards = 10000
# file_centroids = "/pasteur/appa/scratch/rfaure/human-complete.fa"
# input_folder = "/pasteur/appa/scratch/rchikhi/logan_cluster/orfs"
# dict_centroids = "/pasteur/appa/scratch/rfaure/human-complete.tsv" #tab-separated files with two fields: 1/centroid 2/
# input_files = glob.glob(os.path.join(input_folder, "human*.zst"))
# output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins_human/"

num_shards= 100
num_centroids_shards = 10000
file_centroids = "/pasteur/appa/scratch/rfaure/nonhuman_db/centroids.fa"
input_folder = "/pasteur/appa/scratch/rfaure/orfs"
dict_centroids = "/pasteur/appa/scratch/rfaure/nonhuman-complete_tsv_sorted/nonhuman-complete.tsv" #tab-separated files with two fields: 1/centroid 2/
input_files = glob.glob(os.path.join(input_folder, "nonhuman*.zst"))
output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins3/"

# num_shards= 2
# num_centroids_shards = 2
# input_folder = "/pasteur/appa/scratch/rfaure/all_prots_test"
# input_files = glob.glob(os.path.join(input_folder, "*.zst"))
# file_centroids =  "/pasteur/appa/scratch/rfaure/all_prots_test/centroids.fa"
# dict_centroids = "/pasteur/appa/scratch/rfaure/all_prots_test/centroid.tsv"
# output_dir = "/pasteur/appa/scratch/rfaure/all_prots_test/proteins/"

# print("Parameters:")
# print(f"num_shards = {num_shards}")
# print(f"num_centroids_shards = {num_centroids_shards}")
# print(f"file_centroids = {file_centroids}")
# print(f"input_folder = {input_folder}")
# print(f"dict_centroids = {dict_centroids}")
# print(f"input_files = glob.glob(os.path.join(input_folder, \"nonhuman*.zst\"))")
# print(f"output_dir = {output_dir}")

tmp_dir = output_dir + "tmp/"

# Create output and tmp directories if they do not exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

# First step: compute hashes of all prots and store dict_centroid file positions in num_shards files
# Use multiprocessing to bypass GIL for true parallel processing

import multiprocessing as mp
from multiprocessing import Queue, Value, Lock as MPLock
import threading

NUM_WORKERS = 10  # Leave some cores for I/O
CHUNK_SIZE = 64 * 1024 * 1024  # Read 64MB chunks

print(f"Using {NUM_WORKERS} worker processes")

# Multiprocessing queues
task_queue = Queue(maxsize=NUM_WORKERS * 2)
result_queue = Queue(maxsize=NUM_WORKERS * 4)

# Shared counter for progress
lines_processed = Value('L', 0)  # unsigned long

def process_chunk(lines_and_pos):
    """Worker function: process a chunk of lines"""
    lines, start_pos = lines_and_pos
    pos = start_pos
    local_count = 0

    
    # Accumulate positions per shard
    shard_data = [[] for _ in range(num_shards)]
    
    for line in lines:
        if not line:
            pos += 1
            continue
        
        tab_idx = line.find(b' ')
        if tab_idx == -1:
            pos += len(line) + 1
            continue
        
        protein = line[tab_idx + 1:]
        hash_index = hash(protein) % num_shards
        
        shard_data[hash_index].append(pos)
        
        pos += len(line) + 1
        local_count += 1
    
    # Convert lists to packed binary
    result = {}
    for shard_idx in range(num_shards):
        if shard_data[shard_idx]:
            result[shard_idx] = struct.pack(f"{len(shard_data[shard_idx])}Q", *shard_data[shard_idx])
    
    return result, local_count

def reader_process(filename):
    """Read file and send chunks to workers"""
    with open(filename, "rb") as f:
        pos = 0
        leftover = b''
        
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            
            chunk = leftover + chunk
            lines = chunk.split(b'\n')
            leftover = lines[-1]
            lines = lines[:-1]
            
            task_queue.put((lines, pos))
            
            for line in lines:
                pos += len(line) + 1
        
        if leftover:
            task_queue.put(([leftover], pos))
    
    # Signal workers to stop
    for _ in range(NUM_WORKERS):
        task_queue.put(None)

def worker_process():
    """Worker process: get chunks, process them, send results"""
    while True:
        task = task_queue.get()
        if task is None:
            result_queue.put(None)
            break
        
        result, count = process_chunk(task)
        print("Putting thinsg in the result queyrc ", len(result))
        result_queue.put((result, count))

def writer_thread(hash_files):
    """Collect results and write to shard files"""
    workers_done = 0
    
    while workers_done < NUM_WORKERS:
        item = result_queue.get()
        
        if item is None:
            workers_done += 1
            continue
        
        shard_results, count = item
        
        # Write to appropriate shard files
        for shard_idx, data in shard_results.items():
            hash_files[shard_idx].write(data)
        
        # Update counter
        with lines_processed.get_lock():
            lines_processed.value += count

def progress_monitor(start_time):
    """Monitor and report progress"""
    last_count = -1
    
    while True:
        time.sleep(5)
        
        current_count = lines_processed.value
        
        if not any(p.is_alive() for p in workers):
            break
        
        elapsed = time.time() - start_time
        lines_per_sec = current_count / elapsed if elapsed > 0 else 0
        total_lines = 137_000_000_000
        remaining = total_lines - current_count
        eta_sec = remaining / lines_per_sec if lines_per_sec > 0 else 0
        eta = datetime.timedelta(seconds=int(eta_sec))
        print(f"Step 1: processed {current_count:,} proteins, elapsed: {elapsed/60:.2f} min, ETA: {eta}, speed: {lines_per_sec:,.0f} lines/sec")
        sys.stdout.flush()
        
        last_count = current_count

# Open output files
hash_files = [open(f"{tmp_dir}hash_positions_{i}.bin", "wb", buffering=16*1024*1024) for i in range(num_shards)]

# Start processes and threads
start_time = time.time()

reader = mp.Process(target=reader_process, args=(dict_centroids,))
workers = [mp.Process(target=worker_process) for _ in range(NUM_WORKERS)]
writer = threading.Thread(target=writer_thread, args=(hash_files,))
monitor = threading.Thread(target=progress_monitor, args=(start_time,), daemon=True)

reader.start()
for w in workers:
    w.start()
writer.start()
monitor.start()

# Wait for completion
reader.join()
for w in workers:
    w.join()
writer.join()

# Close files
for hf in hash_files:
    hf.close()

print(f"Sharded the centroid dict: {lines_processed.value:,} proteins processed in {(time.time()-start_time)/60:.2f} minutes")

