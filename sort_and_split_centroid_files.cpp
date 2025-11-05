#include <iostream>
#include <vector>
#include <unordered_map>
#include <filesystem>
namespace fs = std::filesystem;

#include <fstream>
#include <sstream>
#include <mutex>
#include <map>
#include <atomic>
#include <chrono>
#include <thread>
#include <future>
#include <random>

using namespace::std;

// simple string hash (explicitely code it so that we can easily export it to pyton)
uint32_t simple_string_hash(const std::string& s) {
    const uint32_t fnv_prime = 16777619U;
    const uint32_t offset_basis = 2166136261U;
    uint32_t hash = offset_basis;
    for (char c : s) {
        hash ^= static_cast<uint32_t>(c);
        hash *= fnv_prime;
    }
    return hash;
}


void sort_and_split_centroid_file(std::string shard_idx, std::string output_dir){
    string file_to_split = output_dir + "tmp/centroid_"+shard_idx+".fa";

    std::string sort_cmd = "awk 'BEGIN{OFS=\"\\t\"}/^>/{if(seq!=\"\")print header,seq;header=$0;seq=\"\";next}NF{if(seq==\"\")seq=$0}END{if(seq!=\"\")print header,seq}' " + file_to_split +
        " | grep -A1 --no-group-separator '^>' | paste - - | sort -k1,1 | awk '!seen[$0]++' | tr '\\t' '\\n' > " + file_to_split + ".sorted";
    int sort_ret = std::system(sort_cmd.c_str());
    if (sort_ret != 0) {
        std::cerr << "ERROR: Failed to sort " << file_to_split << std::endl;
        return;
    }
    file_to_split = file_to_split + ".sorted";

    // cout << " splitting " << file_to_split << endl;
    std::string remove_dir_cmd = "rm " + output_dir + "centroid_" + shard_idx + "/*";
    std::system(remove_dir_cmd.c_str());
    fs::create_directory(output_dir + "centroid_" + shard_idx);

    std::ifstream infile(file_to_split);
    if (!infile) {
        std::cerr << "ERROR: Cannot open " << file_to_split << std::endl;
        return;
    }

    std::vector<std::ofstream> outfiles(1000);
    for (int i = 0; i < 1000; ++i) {
        std::string outname = output_dir + "centroid_" + shard_idx+ "/" + std::to_string(i) + ".fa";
        outfiles[i].open(outname, std::ios::out);
        if (!outfiles[i]) {
            std::cerr << "ERROR: Cannot create " << outname << std::endl;
            return;
        }
    }

    std::ifstream infile_dec(file_to_split);
    if (!infile_dec) {
        std::cerr << "ERROR: Cannot open decompressed file " << file_to_split << std::endl;
        return;
    }

    std::string line, header, sequence;
    while (std::getline(infile_dec, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            header = line;
            if (!std::getline(infile_dec, sequence)){
                break;
            }
            std::string centroid_name = header.substr(1);
            size_t space_pos = centroid_name.find(' ');
            if (space_pos != std::string::npos){
                centroid_name = centroid_name.substr(0, space_pos);
            }
            int hash_val = (simple_string_hash(centroid_name) / 10000) % 1000;
            // if (centroid_name == "SRR5325889_239_2"){
            //     cout << "sequences " << header << " goes with " << sequence.substr(0,10) << "..." << endl;
            // }
            outfiles[hash_val] << header << "\n" << sequence << "\n";
        }
    }

    infile_dec.close();

    for (auto& f : outfiles) f.close();

    std::string compress_cmd = "zstd -f " + output_dir + "centroid_" + shard_idx + "/*.fa";
    auto ret = std::system(compress_cmd.c_str());
    if (ret != 0) {
        std::cerr << "ERROR: Failed to zstd compress files in centroid_" << shard_idx << std::endl;
    }

    // Remove the non-compressed files
    std::string remove_cmd = "rm " + output_dir + "centroid_" + shard_idx + "/*.fa";
    ret = std::system(remove_cmd.c_str());
    if (ret != 0) {
        std::cerr << "ERROR: Failed to remove non-compressed files in centroid_" << shard_idx << std::endl;
    }

    std::string remove_sorted_cmd = "rm " + file_to_split;
    ret = std::system(remove_sorted_cmd.c_str());
    if (ret != 0) {
        std::cerr << "ERROR: Failed to remove sorted file " << file_to_split << std::endl;
    }
}

int main(int argc, char* argv[]) {

    // sort_and_split_centroid_file("3218", "/pasteur/appa/scratch/rfaure/all_prots/proteins3/");
    // exit(1);

    if (argc != 2) {
        std::cerr << "Usage: sort_and_split_centroid_files <shard_idx>" << std::endl;
        return 1;
    }

    int shard_idx;
    try {
        shard_idx = std::stoi(argv[1]);
    } catch (const std::exception&) {
        std::cerr << "ERROR: shard_idx must be an integer" << std::endl;
        return 1;
    }

    const int num_shards = 100;
    int num_threads = 24;
    int num_centroids_shards = 10000;
    const std::string input_folder = "/pasteur/appa/scratch/rchikhi/logan_cluster/orfs";
    const std::string dict_centroids = "/pasteur/appa/scratch/rfaure/human-complete.tsv";
    const std::string output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins_human/";

    // const int num_shards = 100;
    // int num_threads = 24;
    // int num_centroids_shards = 10000;
    // const std::string input_folder = "/pasteur/appa/scratch/rfaure/orfs";
    // const std::string dict_centroids = "/pasteur/appa/scratch/rfaure/nonhuman-complete.tsv";
    // const std::string output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins3/";

    //// Test configuration for all_prots_test
    // const int num_shards = 2;
    // int num_threads = 10;
    // int num_centroids_shards = 2;
    // const std::string file_centroids = "/pasteur/appa/scratch/rfaure/all_prots_test/centroids.fa";
    // const std::string input_folder = "/pasteur/appa/scratch/rfaure/all_prots_test";
    // const std::string dict_centroids = "/pasteur/appa/scratch/rfaure/all_prots_test/centroid.tsv";
    // const std::string output_dir = "/pasteur/appa/scratch/rfaure/all_prots_test/proteins/";

    for (int i = 100*shard_idx ; i<= 100*(shard_idx+1) ; i++){
        sort_and_split_centroid_file(std::to_string(i), output_dir);
    }
  
    cout << "FINISHED SHARD " << shard_idx << endl;

}