#!/usr/bin/env python3
"""
Embed query sequences and save embeddings to a file.
"""

from create_database import embed_glm2_parallel
import argparse
import torch
import numpy as np
import time

__version__ = "1.1.1"


def embed_query_sequences(query_sequences, gpus_available=True, batch_size=10):
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
    for i in range(0, len(query_sequences), batch_size):
        batch = query_sequences[i:i+batch_size]
        if gpus_available:
            try:
                embeddings = embed_glm2_parallel((batch, 0))
            except RuntimeError as e:
                print(f"GPU embedding failed with error: {e}. Falling back to CPU.")
                embeddings = embed_glm2_parallel((batch, "cpu"))
        else:
            embeddings = embed_glm2_parallel((batch, "cpu"))
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
    parser.add_argument("--output", "-o", required=True, help="Path to output embeddings file (.npy)")
    parser.add_argument("--force_cpu", action="store_true", help="Force the use of CPU even if GPUs are available.")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for embedding.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    
    args = parser.parse_args()
    
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
    
    print(f"Reading query sequences from {args.query_sequences}")
    with open(args.query_sequences, "r") as query_file:
        for line in query_file:
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
    
    # Save embeddings
    np.save(args.output, query_embeddings)
    print(f"Embeddings saved to {args.output}")
    
    # Save query names alongside
    names_file = args.output.replace('.npy', '.names.txt')
    with open(names_file, 'w') as f:
        for name in query_names:
            f.write(f"{name}\n")
    print(f"Query names saved to {names_file}")
