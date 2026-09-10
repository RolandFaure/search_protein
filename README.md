# search_protein

A fast protein sequence search tool using embedding-based similarity search. The tool searches for homologs of a query protein through an embedding-based search engine, and as a last steps aligns the results to the query using Mmseqs2 to provide the user sequence alignments. 

---

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

#### A) Conda 
Create a conda environment with all dependencies:

```bash
# Create and activate environment
conda create -n search_fasta -c pytorch -c nvidia -c conda-forge -c bioconda python=3.10 mmseqs2 "pytorch>=2.5" pytorch-cuda=12.1   "faiss-cpu>=1.8" numpy scikit-learn transformers einops
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

#### B) Docker 

```bash
docker pull quay.io/bgruening/logan-protein
```

---

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
                        Path to the output folder (created by embed_query.py if embedding step was run separately)
  --db-type {faiss,usearch}
                        Database type to use: faiss or usearch (default: usearch)
  --outfmt OUTFMT       Format of the mmseqs2 output [0], default is 0 which is a tabular format with header. See mmseqs2 documentation for details.
  -m MEMORY, --memory MEMORY
                        Maximum memory available in GB (mandatory)
  -t NUM_THREADS, --num_threads NUM_THREADS
                        Maximum number of threads available (mandatory)
  --force_cpu           Force the use of CPU even if GPUs are available (for embedding step).
  --version             show program's version number and exit
```

**Output files:**
```
results_folder/
├── PLM_aligned_proteins.tsv    # Set of proteins related to you queries according to the gLM2 protein language model (Logan50 proteins)
├── aligned_proteins.fasta  # Set of proteins aligning on your queries according to mmseqs2
├── aligned_proteins.mmseqs2  # Detail of the mmseqs2 alignments
├── all_proteins.fasta.gz  # All the proteins in Logan corresponding to the PLM aligned proteins in Logan50
└── intermediate_files/
    ├── query_embeddings.npy
    ├── query_embeddings.names.txt
    ├── query_results_intermediate.fasta
    ├── query_results.tsv
    ├── unique_centroids.fasta
    └── matches.top_hit
```

## Main Output Files

- **`PLM_aligned_proteins.tsv`**: TSV files containing Logan50 proteins which gLM2 embeddings have a cosine distance of less 0.2 to the embeddings fo the query.
```
#query_name     result_name     result_sequences        cosine_distance
alpha      ERR11474596_7103_1      MLDWNTSSDIFVEKLLQRNYKSQSLHSQPRHRPQVDGIPYEFGYKGTIYPMNKSRNCIIILLLIPVLVHSTRNAAYFESLEMKIVEQVKLNRAQGKWQLVRELLGLKGTFLKPRWQHFAKTVSSRDFFGNWLPLMLEIERYLYSKKMYPDSYLSWDDHSSYRVRKKVYRRGYDDKGSRRPEHKWFPENAFSRELLDEVPVKRAVYKPFELYHGYSEKRRRRSSLSSLLDL* 0.015888094902038574
```
- **`all_proteins.fasta.gz`**: All Logan proteins found in the Logan50 clusters described in `PLM_aligned_proteins.tsv`.
- **`aligned_proteins.mmseqs2`**: MMseqs2 alignment of proteins aligned against the query (many of the proteins in `all_proteins.fasta.gz` actually do not align to the query using mmseqs)
- **`aligned_proteins.fasta`**: FASTA file containing proteins that align on the query
```
#target  query  identity        alignment_length        nb_mismatches   nb_gap_openings target_start     target_end       query_start    query_end      evalue  bitscore
SRR21885923_17279_1#87#695#-1   alpha   0.924   202     15      0       1       202     1       202     4.604E-129      393
```

## Troubleshooting

### Out of memory
The software automatically adjusts the number of threads used based on the memory available. This optimization is thought for *single-sequence queries*. This can typically lead to out-of-memory errors if querying several proteins.

In case of out-of-memory errors, decrease `-m` below the actual amount of available RAM, or decrease the number of available threads.

### Problem loading the model
The script connects to the internet to load the gLM2 model the first time it runs. Make sure you have an internet connection.

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See LICENSE file for details.
