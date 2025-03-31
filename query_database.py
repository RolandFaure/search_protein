from create_database import embed_glm2_parallel
import faiss
import argparse
import os
import torch

def query_faiss_database(input_fasta, database_folder, query_sequences, cutoff=0.2, num_gpus=1, gpus_available=True):
    """
    Queries a FAISS database with a set of sequences and retrieves the nearest neighbors.

    Args:
        database_folder (str): Path to the folder containing FAISS database files (.bin).
        query_sequences (list of str): List of sequences to query.
        num_gpus (int, optional): Number of GPUs to use for embedding the query sequences. Defaults to 1.

    Returns:
        list of list of tuple: A list where each element corresponds to a query sequence and contains
                                a list of tuples (index, distance) for the nearest neighbors.
    """

    # Embed the query sequences
    print("Embedding query sequences...")
    if gpus_available:
        query_embeddings = embed_glm2_parallel((query_sequences, 0))
    else:
        query_embeddings = embed_glm2_parallel((query_sequences, "cpu"))

    # Load all FAISS indices from the database folder
    print("Querying...")
    results = [[] for _ in query_sequences]
    faiss_indices = []
    for file in os.listdir(database_folder):
        if file.endswith(".bin"):

            file_starting_pos = int(file.strip(".bin").split("_")[2])

            index_path = os.path.join(database_folder, file)
            index = faiss.read_index(index_path)
            nb_of_searches = 1
            distances = []
            while len(distances) == 0 or distances[0][-1] < cutoff or nb_of_searches > 10 : #loop to increase the number of matches if everything matches
                distances, indices = index.search(query_embeddings, k=20*2**(nb_of_searches))
                nb_of_searches+=1

            for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                for dist, idx in zip(dist_row, idx_row):
                    if dist < cutoff:
                        results[i].append((file_starting_pos + idx, dist))

            # sort each row by distance
            print(results, " lsdj ")
            results_sorted = [sorted(result, key=lambda x: x[1]) for result in results]

            # Open the .embeddings file and associate names to distances
            final_results = []
            names_file = os.path.join(database_folder, f"{os.path.basename(input_fasta)}.names")

            with open(names_file, "rb") as nf, open(input_fasta, "r") as fastafile:
                
                for query_idx in range(len(query_sequences)):
                    for result in results_sorted[query_idx] :
                        name_positions = []
                        nf.seek(8*result[0])
                        position_name = int.from_bytes(nf.read(8), byteorder='little', signed=False)

                        fastafile.seek(position_name)
                        name_line = fastafile.readline().strip()
                        sequence_line = fastafile.readline().strip()
                        final_results.append((query_idx, name_line, sequence_line, result[1]))

    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a FAISS database with a sequence.")
    parser.add_argument("--database", required=True, help="Path to the folder containing FAISS database files.")
    parser.add_argument("--input_fasta", required=True, help="Path to the original FASTA file.")
    parser.add_argument("--query_sequences", required=True, help="Fasta file of queris")
    parser.add_argument("--force_cpu", action="store_true", help="Force the use of CPU even if GPUs are available.")
    parser.add_argument("--output_fasta", required=False, help="Path to the output FASTA file. If not provided, results will be printed to stdout.")

    args = parser.parse_args()

    # Check if GPUs are available
    if not args.force_cpu:
        try:
            gpus_available = torch.cuda.is_available()
            print("GPU available, moving on")
        except ImportError:
            gpus_available = False
            print("No GPU available, using CPUs...this could be slow if you have many queries")
    else:
        gpus_available = False

    # Read all sequences from the query FASTA file
    query_sequences = []
    sequence_now = ""
    with open(args.query_sequences, "r") as query_file:
        for line in query_file:
            if line.startswith(">"):
                if sequence_now:
                    query_sequences.append(sequence_now)
                    sequence_now = ""
            else:
                sequence_now += line.strip()
        if sequence_now:
            query_sequences.append(sequence_now)

    query_results = query_faiss_database(
        input_fasta=args.input_fasta,
        database_folder=args.database,
        query_sequences=query_sequences,
        gpus_available=gpus_available
    )

    if args.output_fasta:
        with open(args.output_fasta, "w") as output_file:
            for query, name, sequence, distance in query_results:
                output_file.write(f"{name} ; query {query} Cosine distance: {distance}\n")
                output_file.write(f"{sequence}\n")
    else:
        for name, sequence, distance in query_results:
            print(f"{name} ; Cosine distance: {distance}")
            print(f"{sequence}")
