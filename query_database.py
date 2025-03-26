from create_database import embed_glm2_parallel
import faiss
import argparse
import os

def query_faiss_database(input_fasta, database_folder, query_sequence, cutoff=0.2, num_gpus=1):
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
    query_embeddings = embed_glm2_parallel(([query_sequence], 0))

    # Load all FAISS indices from the database folder
    print("Querying...")
    results = []
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
            results.append([(int(indices[0][j]+file_starting_pos), float(distances[0][j])) for j in range(len(indices[0])) if distances[0][j] < cutoff])

            # Merge and sort all results by distance
            merged_results = []
            for result in results:
                merged_results.extend(result)
            merged_results.sort(key=lambda x: x[1])  # Sort by distance (second element of tuple)

            # Open the .embeddings file and associate names to distances
            final_results = []
            names_file = os.path.join(database_folder, f"{os.path.basename(input_fasta)}.names")

            with open(names_file, "rb") as nf, open(input_fasta, "r") as fastafile:
                
                for result in merged_results :
                    name_positions = []
                    nf.seek(8*result[0])
                    position_name = int.from_bytes(nf.read(8), byteorder='little', signed=False)

                    fastafile.seek(position_name)
                    name_line = fastafile.readline().strip()
                    final_results.append((name_line, result[1]))

    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a FAISS database with a sequence.")
    parser.add_argument("--database", required=True, help="Path to the folder containing FAISS database files.")
    parser.add_argument("--input_fasta", required=True, help="Path to the original FASTA file.")
    parser.add_argument("--query_sequence", required=True, help="Sequence to query the database.")

    args = parser.parse_args()

    query_results = query_faiss_database(
        input_fasta=args.input_fasta,
        database_folder=args.database,
        query_sequence=args.query_sequence
    )
    print(query_results)