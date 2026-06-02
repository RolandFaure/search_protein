# search_protein

A fast protein sequence search tool using embedding-based similarity search with FAISS.

## Overview

`search_fasta` provides a two-step pipeline for searching protein databases:
1. **Embed queries** - Convert protein sequences into embeddings using GLM2
2. **Search database** - Find similar sequences using FAISS index and align with MMseqs2

## Installation

### Requirements

- Python 3.7+
- PyTorch (with CUDA support for GPU acceleration)
- FAISS
- usearch (the vector database), not the DNA/RNA search/clustering tool)
- MMseqs2
- Transformers
- NumPy
- scikit-learn

### Setup

Create a conda environment with all dependencies:

```bash
# Create and activate environment
conda create -n search_fasta create  -c pytorch -c nvidia -c conda-forge -c bioconda   python=3.10 mmseqs2 "pytorch>=2.5" pytorch-cuda=12.1   "faiss-cpu>=1.8" numpy scikit-learn transformers einops
conda activate search_fasta
pip install usearch
```

Then download the repo
```bash
# Clone the repository
git clone https://github.com/RolandFaure/search_protein.git
```
And compile the C++ scripts
```bash
cd search_protein
make
```

## Quick Start

## Search Database

Search your pre-built database with the queries:

```bash
python search_database.py --help
  -h, --help            show this help message and exit
  --query_sequences QUERY_SEQUENCES
                        Fasta file of queries
  --database DATABASE   Path to the folder containing database files.
  --output OUTPUT, -o OUTPUT
                        Path to the output folder (created by embed_query.py if embedding step was run separately).
  --db-type {faiss,usearch}
                        Database type to use: faiss or usearch (default: faiss)
  --outfmt OUTFMT       Format of the mmseqs2 output [0], default is 0 which is a tabular format with header. See mmseqs2
                        documentation for details.
  -m MEMORY, --memory MEMORY
                        Maximum memory available in GB (mandatory)
  -t NUM_THREADS, --num_threads NUM_THREADS
                        Maximum number of threads available (mandatory)
  --force_cpu           Force the use of CPU even if GPUs are available (for embedding step).
  --deep-search         If enabled, extract proteins from all search results instead of only aligned centroids, then align
                        everything with MMseqs2
  --version             show program's version number and exit
```

**Output files:**
```
results_folder/
├── diversified_hits.tsv   # Set of proteins related to you queries according to the gLM2 protein language model
├── matches.fasta          # Set of proteins aligning on your queries according to mmseqs2
├── matches.mmseqs2        # Detail of the mmseqs2 alignments
└── intermediate_files/
    ├── query_embeddings.npy
    ├── query_embeddings.names.txt
    ├── query_results_intermediate.fasta
    ├── query_results.tsv
    ├── unique_centroids.fasta
    ├── all_results.fasta # All proteins bearing some similarities to query
    └── matches.top_hit
```

## Main Output Files

- **`diversified_hits.tsv`**: TSV files containing Logan proteins which gLM2 embeddings have a cosine distance of less 0.2 to the embeddings fo the query.
```
#query_name     result_name     result_sequences        cosine_distance
alpha      ERR11474596_7103_1      MLDWNTSSDIFVEKLLQRNYKSQSLHSQPRHRPQVDGIPYEFGYKGTIYPMNKSRNCIIILLLIPVLVHSTRNAAYFESLEMKIVEQVKLNRAQGKWQLVRELLGLKGTFLKPRWQHFAKTVSSRDFFGNWLPLMLEIERYLYSKKMYPDSYLSWDDHSSYRVRKKVYRRGYDDKGSRRPEHKWFPENAFSRELLDEVPVKRAVYKPFELYHGYSEKRRRRSSLSSLLDL* 0.015888094902038574
```
- **`matches.fasta`**: FASTA file containing all matched protein sequences
- **`matches.mmseqs2`**: MMseqs2 alignment results of matched protein versus queries
```
#target  query  identity        alignment_length        nb_mismatches   nb_gap_openings target_start     target_end       query_start    query_end      evalue  bitscore
SRR21885923_17279_1#87#695#-1   alpha   0.924   202     15      0       1       202     1       202     4.604E-129      393
```

## Performance Tips

To give an order of magnitude, searching for one protein takes 9000 CPU.s on my setup. Here are a few 
key performance points to understand.

1. **Batching queries**: The time needed to search through the database is strongly sub-linear in number of queries.
On my system, the performance went from 2000 CPU.s for one query and 3000 CPU.s for 2000 queries on the nonhuman database.

2. **Parallel Search**: Use multiple processes for faster database search. 
Count ~200M of RAM per thread.

3. **GPU Acceleration**: Use GPU for embedding 
    At embedding time, using GPU takes ~0.1s / query versus 20s / query on CPU.
    Keep in mind that embedding time is not significant for few queries.
   ```bash
   python embed_query.py --query_sequences queries.fasta --output results -F
   # GPU will be used automatically if available, use --force_cpu if you do not want
   ```


## Troubleshooting

### Out of memory

If you encounter out of memory errors during search, reduce the number of threads with `-t` (each process requires ~200MB of RAM, but this might depend on the query)

### Problem loading the model
The script connects to the internet to load the gLM2 model the first time it runs. Make sure you have an internet connection.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See LICENSE file for details.
