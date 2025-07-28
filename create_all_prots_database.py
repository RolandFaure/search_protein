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

if len(sys.argv) != 2:
    print("Usage: python create_all_prots_database.py <shard_idx>")
    sys.exit(1)

try:
    shard_idx = int(sys.argv[1])
except ValueError:
    print("ERROR: shard_idx must be an integer")
    sys.exit(1)

file_locks = {}

num_shards= 100
num_threads = 30
num_centroids_shards = 10000
file_centroids = "/pasteur/appa/scratch/rfaure/human-complete.fa"
input_folder = "/pasteur/appa/scratch/rchikhi/logan_cluster/orfs/"
dict_centroids = "/pasteur/appa/scratch/rfaure/human-complete.tsv" #tab-separated files with two fields: 1/centroid 2/
input_files = glob.glob(os.path.join(input_folder, "human*.zst"))
output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins_human/"

# num_shards= 100
# num_threads = 1
# num_centroids_shards = 10000
# file_centroids = "/pasteur/appa/scratch/rfaure/nonhuman-complete.fa"
# input_folder = "/pasteur/appa/scratch/rchikhi/logan_cluster/orfs/"
# dict_centroids = "/pasteur/appa/scratch/rfaure/nonhuman-complete.tsv" #tab-separated files with two fields: 1/centroid 2/
# input_files = glob.glob(os.path.join(input_folder, "nonhuman*.zst"))
# output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins3/"

# num_shards= 2
# num_threads = 10
# num_centroids_shards = 2
# input_folder = "/pasteur/appa/scratch/rfaure/all_prots_test"
# input_files = glob.glob(os.path.join(input_folder, "*.zst"))
# file_centroids =  "/pasteur/appa/scratch/rfaure/all_prots_test/centroids.fa"
# dict_centroids = "/pasteur/appa/scratch/rfaure/all_prots_test/centroid.tsv"
# output_dir = "/pasteur/appa/scratch/rfaure/all_prots_test/proteins/"

print("Parameters:")
print(f"num_shards = {num_shards}")
print(f"num_threads = {num_threads}")
print(f"num_centroids_shards = {num_centroids_shards}")
print(f"file_centroids = {file_centroids}")
print(f"input_folder = {input_folder}")
print(f"dict_centroids = {dict_centroids}")
print(f"input_files = glob.glob(os.path.join(input_folder, \"human*.zst\"))")
print(f"output_dir = {output_dir}")

tmp_dir = output_dir + "tmp/"

# Create output and tmp directories if they do not exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tmp_dir, exist_ok=True)

# # First step: compute hashes of all prots and store dict_centroid file positions in num_shards files
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

# # Second step: shard each input file in different pieces
# def process_input_file(input_file):
#     base_name = os.path.basename(input_file).replace('.zst', '')
#     shard_files = [open(f"{tmp_dir}{base_name}_shard_{i}.fa", "w") for i in range(num_shards)]
#     nb_lines=  0

#     with open(input_file, "rb") as compressed:
#         dctx = zstd.ZstdDecompressor()
#         with dctx.stream_reader(compressed) as reader:
#             text_reader = io.TextIOWrapper(reader)
#             header = None
#             seq = ""
#             while True:
#                 line = text_reader.readline()
#                 if nb_lines % 100000000 == 0 :
#                     print("in fuile ", input_file, " just sorted ", nb_lines)
#                 nb_lines += 1
#                 if not line:
#                     break
#                 line = line.rstrip('\n')
#                 if line.startswith(">"):
#                     if header is not None:
#                         protein_id = header.split()[0]
#                         shard_idx = hash(protein_id) % num_shards
#                         header_fields = header.split(" ")
#                         new_description = " ".join(header_fields[:7])
#                         record_str = f">{new_description}\n{seq}\n"
#                         shard_files[shard_idx].write(record_str)
#                     header = line[1:]
#                     seq = ""
#                 else:
#                     seq += line
#             # Handle last record
#             if header is not None:
#                 protein_id = header.split()[0]
#                 shard_idx = hash(protein_id) % num_shards
#                 header_fields = header.split(" ")
#                 new_description = " ".join(header_fields[:7])
#                 record_str = f">{new_description}\n{seq}\n"
#                 shard_files[shard_idx].write(record_str)

#     for sf in shard_files:
#         sf.close()

#     # Compress all shard files we just created using zst
#     for i in range(num_shards):
#         shard_file = f"{tmp_dir}{base_name}_shard_{i}.fa"
#         if os.path.exists(shard_file):
#             compressed_file = f"{shard_file}.zst"
#             os.system(f"zstd -3 -f -q {shard_file}")
    
#     print("just sharded ", input_file)

# with ThreadPoolExecutor(max_workers=num_threads) as executor:
#     executor.map(process_input_file, input_files)
# print("sharded the protein files")


# Third step: dispatch proteins in their centroid files

# def process_input_file_for_shard(args):
#     start_time = time.time()
#     input_file, shard_idx, prot_to_centroid_dict = args
#     shard_file = f"{tmp_dir}{os.path.basename(input_file).replace('.zst', '')}_shard_{shard_idx}.fa"
#     if not os.path.exists(shard_file):
#         return

#     # Process each record in the decompressed fasta file
#     with open(shard_file, "r") as fasta_handle:
#         while True:
#             header_line = fasta_handle.readline()
#             if not header_line:
#                 break
#             header_line = header_line.rstrip('\n')
#             if not header_line.startswith(">"):
#                 continue
#             protein_id = header_line[1:].split()[0]
#             if protein_id not in prot_to_centroid_dict:
#                 print("ERROR 330: ", protein_id, " is not in shard ", shard_idx, " but is found in ", shard_file)
#                 continue
#             centroid = prot_to_centroid_dict[protein_id]
#             centroid_hash = hash(centroid) % num_centroids_shards
#             output_file = os.path.join(tmp_dir, f"centroid_{centroid_hash}_tmp.fa")
#             header_fields = header_line[1:].split(" ")
#             new_description = centroid + " " + " ".join(header_fields[:7])
#             sequence = fasta_handle.readline().rstrip('\n')
#             lock = file_locks.setdefault(output_file, Lock())
#             with lock:
#                 with open(output_file, "a") as out_handle:
#                     out_handle.write(f">{new_description}\n{sequence}\n")
#                 if os.path.exists(output_file) and os.path.getsize(output_file) > 100 * 1024 * 1024:
#                     compressed_file = os.path.join(tmp_dir, f"centroid_{centroid_hash}.fa.zst")
#                     cctx = zstd.ZstdCompressor(level=3)
#                     with open(output_file, "rb") as src, open(compressed_file, "ab") as dst:
#                         dst.write(cctx.compress(src.read()))
#                     os.remove(output_file)

#     elapsed = time.time() - start_time
#     print(f"In shard {shard_idx}, just dispatched {input_file} in {elapsed:.2f} seconds")

# def process_shard(shard_idx):
#     # 1. Load dict_centroid dictionary for this shard
#     prot_to_centroid_dict = {}
#     line_positions = []
#     print("Loading dictionnary for shard ", shard_idx)
#     total_positions = os.path.getsize(f"{tmp_dir}hash_positions_{shard_idx}.bin") // 8
#     with open(f"{tmp_dir}hash_positions_{shard_idx}.bin", "rb") as pos_file:
#         count = 0
#         start_time = time.time()
#         while True:
#             bytes_read = pos_file.read(8)
#             if not bytes_read:
#                 break
#             line_positions.append(struct.unpack("Q", bytes_read)[0])
#             count += 1
#             if count % 10_000_000 == 0:
#                 elapsed = time.time() - start_time
#                 lines_per_sec = count / elapsed if elapsed > 0 else 0
#                 remaining = total_positions - count
#                 eta_sec = remaining / lines_per_sec if lines_per_sec > 0 else 0
#                 eta = datetime.timedelta(seconds=int(eta_sec))
#                 print(f"Shard {shard_idx}: loaded {count:,}/{total_positions:,} positions, elapsed: {elapsed/60:.2f} min, ETA: {eta}")
#                 break

#     print("Loaded the positions of the lines, now loading the actual dict")

#     with open(dict_centroids, "r") as f:
#         total_positions = len(line_positions)
#         start_time = time.time()
#         for idx, pos in enumerate(line_positions):
#             f.seek(pos)
#             line = f.readline()
#             if not line:
#                 continue
#             centroid, protein = line.strip().split('\t')
#             prot_to_centroid_dict[protein] = centroid
#             print("censdfq ", centroid, " ", protein)
#             if (idx + 1) % 10_000_000 == 0:
#                 elapsed = time.time() - start_time
#                 lines_per_sec = (idx + 1) / elapsed if elapsed > 0 else 0
#                 remaining = total_positions - (idx + 1)
#                 eta_sec = remaining / lines_per_sec if lines_per_sec > 0 else 0
#                 eta = datetime.timedelta(seconds=int(eta_sec))
#                 print(f"Shard {shard_idx}: loaded {idx + 1:,}/{total_positions:,} dict entries, elapsed: {elapsed/60:.2f} min, ETA: {eta}")

#     print("Loaded the dictionnary")

#     # 2. For this shard, process all input files in parallel
#     args_list = [(input_file, shard_idx, prot_to_centroid_dict) for input_file in input_files]
#     with ThreadPoolExecutor(max_workers=num_threads) as executor:
#         executor.map(process_input_file_for_shard, args_list)

# process_shard(shard_idx)

# # After all shards are processed, integrate any remaining tmp files into the final zstd files
# def sort_output_file(shard_idx):
#     start_time = time.time()

#     input_file = os.path.join(tmp_dir, f"centroid_{shard_idx}.fa")
#     sorted_file = os.path.join(output_dir, f"centroid_{shard_idx}.sorted.fa.zst")
#     command = f"sed '/^*$/d' {input_file} | sed '/^$/d' | grep -A1 --no-group-separator '^>' | paste - - | sort -k1,1 | awk '!seen[$0]++' | tr '\\t' '\\n' | zstd > {sorted_file}"
#     subprocess.run(command, shell=True, check=True)

#     elapsed = time.time() - start_time
#     print(f"sorted and recompressed {sorted_file} in {elapsed:.2f} seconds")

# print("number of cpus: ", num_threads)
# with ProcessPoolExecutor(max_workers=num_threads) as executor:
#     executor.map(sort_output_file, range(shard_idx*100, min(num_centroids_shards, shard_idx*100+100)))
# print("Sorted all centroid output files by record name")

# def create_centroid_index(sorted_file):
#     index = {}
#     position = 0
#     with open(sorted_file, "rb") as f:
#         dctx = zstd.ZstdDecompressor()
#         with dctx.stream_reader(f) as reader:
#             text_reader = io.TextIOWrapper(reader)
#             while True:
#                 line = text_reader.readline()
#                 if not line:
#                     break
#                 if line.startswith(">"):
#                     centroid = line[1:].split()[0]
#                     if centroid not in index:
#                         index[centroid] = position
#                 position = reader.tell()
#     return index

# def index_shard(shard):
#     sorted_file = os.path.join(output_dir, f"centroid_{shard}.sorted.fa.zst")
#     if not os.path.exists(sorted_file):
#         return
#     print(f"Indexing {sorted_file}")
#     index = create_centroid_index(sorted_file)

#     idx_file = os.path.join(output_dir, f"centroid_{shard}.index.tsv")
#     with open(idx_file, "w") as fidx:
#         for centroid, pos in index.items():
#             fidx.write(f"{centroid}\t{pos}\n")

#     print(f"Saved index to {pkl_file}")


# with ThreadPoolExecutor(max_workers=num_threads) as executor:
#     executor.map(index_shard, range(shard_idx*50, min(num_centroids_shards, shard_idx*50+50)))




