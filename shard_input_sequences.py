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

print("got: ", " ".join(sys.argv))
sys.stdout.flush()
if len(sys.argv) != 3:
    print("Usage: python create_all_prots_database.py <shard_idx> <num_parallel_jobs>")
    sys.exit(1)

file_locks = {}
global_shard_idx = int(sys.argv[1])
num_parallel_jobs = int(sys.argv[2])

# num_shards= 100
# num_threads = 1
# num_centroids_shards = 10000
# file_centroids = "/pasteur/appa/scratch/rfaure/human-complete.fa"
# input_folder = "/pasteur/appa/scratch/rchikhi/logan_cluster/orfs/"
# dict_centroids = "/pasteur/appa/scratch/rfaure/human-complete.tsv" #tab-separated files with two fields: 1/centroid 2/
# input_files = glob.glob(os.path.join(input_folder, "human*.zst"))
# output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins_human/"

num_shards= 100
num_threads = 1
num_centroids_shards = 10000
file_centroids = "/pasteur/appa/scratch/rfaure/nonhuman_db/centroids.fa"
input_folder = "/pasteur/appa/scratch/rfaure/orfs"
dict_centroids = "/pasteur/appa/scratch/rfaure/nonhuman-complete_tsv_sorted/nonhuman-complete.tsv" #tab-separated files with two fields: 1/centroid 2/
input_files = glob.glob(os.path.join(input_folder, "nonhuman*.zst"))
output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins3/"


# num_shards= 2
# num_threads = 10
# num_centroids_shards = 2
# input_folder = "/pasteur/appa/scratch/rfaure/all_prots_test"
# input_files = glob.glob(os.path.join(input_folder, "*.zst"))
# file_centroids =  "/pasteur/appa/scratch/rfaure/all_prots_test/centroids.fa"
# dict_centroids = "/pasteur/appa/scratch/rfaure/all_prots_test/centroid.tsv"
# output_dir = "/pasteur/appa/scratch/rfaure/all_prots_test/proteins/"

# print("Parameters:")
# print(f"num_shards = {num_shards}")
# print(f"num_threads = {num_threads}")
# print(f"num_centroids_shards = {num_centroids_shards}")
# print(f"file_centroids = {file_centroids}")
# print(f"input_folder = {input_folder}")
# print(f"dict_centroids = {dict_centroids}")
# print(f"input_files = glob.glob(osvim.path.join(input_folder, \"nonhuman*.zst\"))")
# print(f"output_dir = {output_dir}")

tmp_dir = output_dir + "tmp/"

# Create output and tmp directories if they do not exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

# Second step: shard each input file in different pieces
def process_input_file(input_file):
    base_name = os.path.basename(input_file).replace('.zst', '')
    shard_files = [open(f"{tmp_dir}{base_name}_shard_{i}.fa", "w") for i in range(num_shards)]
    nb_lines=  0

    with open(input_file, "rb") as compressed:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed) as reader:
            text_reader = io.TextIOWrapper(reader)
            header = None
            seq = ""
            while True:
                line = text_reader.readline()
                if nb_lines % 100000000 == 0 :
                    print("in fuile ", input_file, " just sorted ", nb_lines)
                nb_lines += 1
                if not line:
                    break
                line = line.rstrip('\n')
                if line.startswith(">"):
                    if header is not None:
                        protein_id = header.split()[0]
                        shard_idx = hash(protein_id) % num_shards
                        header_fields = header.split(" ")
                        new_description = " ".join(header_fields[:7])
                        record_str = f">{new_description}\n{seq}\n"
                        shard_files[shard_idx].write(record_str)
                    header = line[1:]
                    seq = ""
                else:
                    seq += line
            # Handle last record
            if header is not None:
                protein_id = header.split()[0]
                shard_idx = hash(protein_id) % num_shards
                header_fields = header.split(" ")
                new_description = " ".join(header_fields[:7])
                record_str = f">{new_description}\n{seq}\n"
                shard_files[shard_idx].write(record_str)

    for sf in shard_files:
        sf.close()

    # Compress all shard files we just created using zst
    for i in range(num_shards):
        shard_file = f"{tmp_dir}{base_name}_shard_{i}.fa"
        if os.path.exists(shard_file):
            compressed_file = f"{shard_file}.zst"
            os.system(f"zstd -3 -f -q {shard_file}")
            os.remove(shard_file)
    
    print("just sharded ", input_file)

# Filter input files based on the global shard index and number of parallel jobs
input_files = [f for f in input_files if (hash(os.path.basename(f)) % num_parallel_jobs) == global_shard_idx]
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    executor.map(process_input_file, input_files)
print("sharded the protein files ", global_shard_idx, " / ", num_parallel_jobs)




