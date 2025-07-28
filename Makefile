all:
	g++ -std=c++17 -o create_all_prots_database create_all_prots_database.cpp -lstdc++fs -lpthread
	g++ -std=c++17 -o centroid_to_prots centroid_to_prots.cpp -lstdc++fs -lpthread -lzstd