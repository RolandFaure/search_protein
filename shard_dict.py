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

num_shards= 100
num_centroids_shards = 10000
file_centroids = "/pasteur/appa/scratch/rfaure/human-complete.fa"
input_folder = "/pasteur/appa/scratch/rchikhi/logan_cluster/orfs"
dict_centroids = "/pasteur/appa/scratch/rfaure/human-complete.tsv" #tab-separated files with two fields: 1/centroid 2/
input_files = glob.glob(os.path.join(input_folder, "human*.zst"))
output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins_human/"

# num_shards= 100
# num_centroids_shards = 10000
# file_centroids = "/pasteur/appa/scratch/rfaure/nonhuman-complete.fa"
# input_folder = "/pasteur/appa/scratch/rfaure/orfs"
# dict_centroids = "/pasteur/appa/scratch/rfaure/nonhuman-complete.tsv" #tab-separated files with two fields: 1/centroid 2/
# input_files = glob.glob(os.path.join(input_folder, "nonhuman*.zst"))
# output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins3/"

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
hash_files = [open(f"{tmp_dir}hash_positions_{i}.bin", "wb") for i in range(num_shards)]

line_nb = 0
start_time = time.time()
with open(dict_centroids, "r") as f:
   pos = 0
   while True:
       line = f.readline()
       if not line:
           break
       centroid, protein = line.strip().split('\t')
       hash_index = hash(protein) % num_shards
       # Store the file offset as int64 in the corresponding file
       hash_files[hash_index].write(struct.pack("Q", pos))  # use 8 bytes for file offset
       pos += len(line)
       if line_nb % 1000000 == 0 and line_nb > 0:
           elapsed = time.time() - start_time
           lines_per_sec = line_nb / elapsed if elapsed > 0 else 0
           total_lines = 137_000_000_000
           remaining = total_lines - line_nb
           eta_sec = remaining / lines_per_sec if lines_per_sec > 0 else 0
           eta = datetime.timedelta(seconds=int(eta_sec))
           print(f"Step 1: processed {line_nb:,} proteins, elapsed: {elapsed/60:.2f} min, ETA: {eta}")
           sys.stdout.flush()
       line_nb += 1

for hf in hash_files:
   hf.close()

print("sharded the centroid dict")

