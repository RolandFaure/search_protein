all:
	g++ -std=c++17 -o create_all_prots_database create_all_prots_database.cpp -lstdc++fs -lpthread
	g++ -std=c++17 -o centroid_to_prots centroid_to_prots.cpp -lstdc++fs -lpthread -lzstd
	g++ -std=c++17 -o correct_missing_linebreaks correct_missing_linebreaks.cpp -lstdc++fs -lpthread
	g++ -std=c++17 -o sort_centroid_files sort_centroid_files.cpp -lstdc++fs -lpthread
	g++ -std=c++17 -o create_gene_matrix create_gene_matrix.cpp -lstdc++fs -lpthread