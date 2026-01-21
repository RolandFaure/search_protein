#!/usr/bin/env python3
"""
Embed query sequences and save embeddings to a file.
"""

from create_database import embed_glm2_parallel
import argparse
import torch
import numpy as np
import time
import subprocess
import tempfile
import os
import shutil

__version__ = "1.2.0"


def embed_query_sequences(query_sequences, gpus_available=True, batch_size=20):
    """
    Embed query sequences using GLM2 model.
    
    Args:
        query_sequences (list of str): List of sequences to embed.
        gpus_available (bool): Whether GPUs are available.
        batch_size (int): Batch size for embedding.
    
    Returns:
        np.ndarray: Embeddings as a numpy array.
    """
    print("Embedding query sequences...")
    start_time = time.time()
    
    query_embeddings_list = []
    total_batches = (len(query_sequences) + batch_size - 1) // batch_size
    batch_times = []
    
    for batch_idx, i in enumerate(range(0, len(query_sequences), batch_size)):
        batch_start = time.time()
        batch = query_sequences[i:i+batch_size]
        if gpus_available:
            try:
                embeddings = embed_glm2_parallel((batch, 0))
            except RuntimeError as e:
                print(f"GPU embedding failed with error: {e}. Falling back to CPU.")
                embeddings = embed_glm2_parallel((batch, "cpu"))
        else:
            embeddings = embed_glm2_parallel((batch, "cpu"))

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        
        # Calculate ETA
        avg_batch_time = np.mean(batch_times[-10:]) if len(batch_times) > 0 else batch_time
        remaining_batches = total_batches - (batch_idx + 1)
        eta_seconds = avg_batch_time * remaining_batches
        eta_minutes = eta_seconds / 60
        
        print(f"Embedded batch {batch_idx + 1}/{total_batches} ({i+batch_size} proteins) - ETA: {eta_minutes:.1f} min")
        query_embeddings_list.append(embeddings)
    
    query_embeddings = np.concatenate(query_embeddings_list, axis=0)
    
    # Ensure query_embeddings is a CPU numpy array
    if hasattr(query_embeddings, "cpu"):
        query_embeddings = query_embeddings.cpu()
    if hasattr(query_embeddings, "numpy"):
        query_embeddings = query_embeddings.numpy()
    
    elapsed_time = time.time() - start_time
    print(f"Embedding completed in {elapsed_time:.2f} seconds")
    
    return query_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed query sequences and save to file.")
    parser.add_argument("--query_sequences", required=True, help="Fasta file of queries")
    parser.add_argument("--output", "-o", required=True, help="Path to output folder")
    parser.add_argument("-F", "--force", action="store_true", help="Force cleaning the output folder if it exists")
    parser.add_argument("--force_cpu", action="store_true", help="Force the use of CPU even if GPUs are available.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for embedding.")
    parser.add_argument("-r","--reduce_query", action="store_true", help="Cluster similar proteins to reduce embedding time (identity > 0.9)")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    args = parser.parse_args()
    
    # Handle output folder creation/cleanup
    output_folder = args.output.rstrip("/")
    if os.path.exists(output_folder):
        if args.force:
            print(f"Cleaning existing output folder: {output_folder}")
            shutil.rmtree(output_folder)
            os.makedirs(output_folder)
        else:
            print(f"Error: Output folder '{output_folder}' already exists. Use -F/--force to overwrite.")
            exit(1)
    else:
        os.makedirs(output_folder)
    
    # Create intermediate_files subfolder
    intermediate_folder = os.path.join(output_folder, "intermediate_files")
    os.makedirs(intermediate_folder, exist_ok=True)
    
    # Check if GPUs are available
    if not args.force_cpu:
        try:
            gpus_available = torch.cuda.is_available()
            if gpus_available:
                print("GPU available")
            else:
                print("No GPU available, using CPU")
        except ImportError:
            gpus_available = False
            print("No GPU available, using CPU")
    else:
        gpus_available = False
        print("Forced CPU mode")
    
    # Read all sequences from the query FASTA file
    query_sequences = []
    query_names = []
    sequence_now = ""
    query_file = args.query_sequences

    if args.reduce_query:
            
        print("Clustering similar proteins with MMseqs2...")
        

        # Run MMseqs2 easy-linclust in intermediate folder
        cluster_prefix = os.path.join(intermediate_folder, "tmp_query_cluster")
        rep_fasta = os.path.join(intermediate_folder, "tmp_cluster_rep_seq.fasta")
        
        try:
            subprocess.run([
                "mmseqs", "easy-linclust",
                query_file,
                cluster_prefix,
                intermediate_folder,
                "--min-seq-id", "0.9",
                "-c", "0.8",
                "--cov-mode", "1"
            ], check=True, capture_output=True, text=True)
            
            # Use the representative sequences file as the query file
            query_file = f"{cluster_prefix}_rep_seq.fasta"
            print(f"Using clustered representative sequences: {query_file}")
            
        except subprocess.CalledProcessError as e:
            print(f"MMseqs2 clustering failed: {e}")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Error output: {e.stderr}")
            print("Proceeding without clustering")
        except FileNotFoundError:
            print("MMseqs2 not found. Please install MMseqs2 to use --reduce_query option.")
            print("Proceeding without clustering")
    
    print(f"Reading query sequences from {args.query_sequences}")
    with open(query_file, "r") as query_file_s:
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
    
    # Embed the sequences
    query_embeddings = embed_query_sequences(
        query_sequences=query_sequences,
        gpus_available=gpus_available,
        batch_size=args.batch_size
    )
    
    # Save embeddings in intermediate_files
    embeddings_file = os.path.join(intermediate_folder, "query_embeddings.npy")
    np.save(embeddings_file, query_embeddings)
    print(f"Embeddings saved to {embeddings_file}")
    
    # Save query names in intermediate_files
    names_file = os.path.join(intermediate_folder, "query_embeddings.names.txt")
    with open(names_file, 'w') as f:
        for name in query_names:
            f.write(f"{name}\n")
    print(f"Query names saved to {names_file}")

    
    # Clean up temporary files from MMseqs2 clustering (already in intermediate_folder, no action needed)
    # The files are created in intermediate_folder, so they'll stay there
