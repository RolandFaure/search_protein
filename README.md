# search_fasta

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

```

Then download the repo
```bash
# Clone the repository
git clone https://github.com/RolandFaure/search_protein.git
```


## Quick Start

### Step 1: Embed Query Sequences

Convert your query sequences into embeddings:

```bash
python embed_query.py \
    --query_sequences queries.fasta \
    --output results_folder \
    -F
```

**Parameters:**
- `--query_sequences`: Input FASTA file with query sequences
- `--output`, `-o`: Output folder (will be created)
- `-F`, `--force`: Force overwrite if output folder exists
- `--force_cpu`: Force CPU usage even if GPU is available

**Output structure:**
```
results_folder/
├── intermediate_files/
│   ├── query_embeddings.npy
│   ├── query_embeddings.names.txt
│   └── [MMseqs2 clustering files if -r used]
```

### Step 2: Search Database

Search your pre-built FAISS database with the embedded queries:

```bash
python search_database.py \
    --database /path/to/database \
    --output results_folder \
    --query_sequences queries.fasta \
    -t 8
```

**Parameters:**
- `--database`: Path to the FAISS database folder
- `--output`, `-o`: Output folder (same as from embed_query.py)
- `--query_sequences`: Original query FASTA file (for MMseqs2 alignment)
- `-t`, `--num_threads`: Number of threads (default: 1)
- `--outfmt`: Output format for MMseqs2 (default: '0' for tabular with header)

**Output files:**
```
results_folder/
├── matches.fasta          # Main output: matched proteins
├── matches.mmseqs2        # MMseqs2 alignment results
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

- **`matches.fasta`**: FASTA file containing all matched protein sequences
- **`matches.mmseqs2`**: MMseqs2 alignment results of matched protein versus queries

## Complete Example

```bash
# 1. Embed your queries (with GPU acceleration and query reduction)
python embed_query.py \
    --query_sequences my_proteins.fasta \
    --output search_results \
    -F

# 2. Search the database
python search_database.py \
    --database /data/protein_database \
    --output search_results \
    --query_sequences my_proteins.fasta \
    -t 16

# 3. View results
less search_results/matches.mmseqs2
```

## Performance Tips

1. **GPU Acceleration**: Use GPU for embedding (~1s / query, vs 120s / query on CPU)
   ```bash
   python embed_query.py --query_sequences queries.fasta --output results -F
   # GPU will be used automatically if available
   ```

2. **Parallel Search**: Use multiple threads for faster database search. */!\ RAM consumption of 20GB / process*
   ```bash
   python search_database.py --database db --output results --query_sequences queries.fasta -t 32
   ```

## Troubleshooting

### Out of memory

If you encounter out of memory errors during search, reduce the number of threads with `-t` (each process requires ~25GB of RAM)

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See LICENSE file for details.