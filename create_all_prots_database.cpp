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
#include <algorithm>

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

// prot_to_centroid_dict: unordered_map<string, string>
void process_input_file_for_shard(
    const std::string& input_file,
    int shard_idx,
    const std::unordered_map<std::string, std::string>& prot_to_centroid_dict,
    std::string tmp_dir)
{
    auto start_time = std::chrono::steady_clock::now();
    std::string base = fs::path(input_file).filename().string();
    std::string shard_file = tmp_dir + base.substr(0,base.size()-4) + "_shard_" + std::to_string(shard_idx) + ".fa.zst";
    if (!fs::exists(shard_file)) {
        cout << "shard file "<< shard_file << " does not exist, return code 4510" << endl;
        return;
    }

    // Decompress the zstd file
    std::string decompressed_file = shard_file + ".decompressed";
    if (!fs::exists(decompressed_file)) {
        std::string decompress_cmd = "zstd -d -c " + shard_file + " > " + decompressed_file;
        int decompress_result = system(decompress_cmd.c_str());
        if (decompress_result != 0) {
            std::cerr << "ERROR: Failed to decompress " << shard_file << std::endl;
            return;
        }
    }
    shard_file = decompressed_file;

    // Create index file for this input file
    std::string index_file = tmp_dir + "index_" + base.substr(0,base.size()-4) + "_shard_" + std::to_string(shard_idx) + ".bin";
    std::ofstream index_out(index_file, std::ios::binary);
    if (!index_out) {
        std::cerr << "ERROR: Cannot create index file " << index_file << std::endl;
        return;
    }

    std::string decompressed_file_with_centroid = shard_file + ".decompressed.centroid.fa";
    std::ofstream fasta_out(decompressed_file_with_centroid);

    int number_of_missed_proteins = 0;
    int number_of_dispatched_proteins = 0;

    std::ifstream fasta_handle(shard_file);
    if (!fasta_handle) return;
    
    std::string header_line;
    uint64_t centroid_file_offset = 0;
    
    while (fasta_handle.tellg() != -1) {
        
        if (!std::getline(fasta_handle, header_line)) break;
        if (header_line.empty() || header_line[0] != '>') continue;
        
        std::string protein_id = header_line.substr(1);
        size_t space_pos = protein_id.find(' ');
        if (space_pos != std::string::npos){
            protein_id = protein_id.substr(0, space_pos);
        }

        auto it = prot_to_centroid_dict.find(protein_id);
        if (it == prot_to_centroid_dict.end()) {
            if (number_of_missed_proteins <= 10){
                std::cerr << "ERROR 330: " << protein_id << " is not in shard " << shard_idx
                        << " but is found in " << shard_file << std::endl;
            }
            number_of_missed_proteins += 1;

            
            // Skip sequence line
            std::string sequence;
            std::getline(fasta_handle, sequence);
        }
        else{
            const std::string& centroid = it->second;
            auto centroid_hash = simple_string_hash(centroid);
            int dir_hash = centroid_hash % 10000;
            int file_hash = (centroid_hash / 10000) % 1000;
            
            // Write index entry: dir_hash (4 bytes), file_hash (4 bytes), offset in centroid file (8 bytes)
            index_out.write(reinterpret_cast<const char*>(&dir_hash), sizeof(int));
            index_out.write(reinterpret_cast<const char*>(&file_hash), sizeof(int));
            index_out.write(reinterpret_cast<const char*>(&centroid_file_offset), sizeof(uint64_t));
            
            number_of_dispatched_proteins += 1;
            std::string new_header = ">" + centroid + "_" + header_line.substr(1) + "\n";
            fasta_out << new_header;
            centroid_file_offset += new_header.length();
            
            // Skip sequence line
            std::string sequence;
            std::getline(fasta_handle, sequence);
            fasta_out << sequence << "\n";
            centroid_file_offset += sequence.length() + 1;
        }
    }

    // Replace the decompressed file with the centroid version
    fasta_out.close();
    fasta_handle.close();
    std::remove(shard_file.c_str());
    std::rename(decompressed_file_with_centroid.c_str(), shard_file.c_str());

    index_out.close();

    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "In shard " << shard_idx << ", just indexed " << shard_file << " to " << index_file
              << " in " << elapsed << " seconds. In total, indexed " << number_of_dispatched_proteins << " and did not find " << number_of_missed_proteins << " proteins." << std::endl;
}

/**
 * @brief Processes a shard of protein data by loading centroid mappings and processing input files in parallel.
 *
 * This function performs the following steps:
 * 1. Loads line positions from a binary file corresponding to the shard.
 * 2. Loads a dictionary mapping proteins to centroids using the loaded positions.
 * 3. Processes all input files assigned to the shard in parallel using multiple threads.
 *
 * @param shard_idx Index of the current shard being processed.
 * @param dict_centroids Path to the file containing centroid dictionary entries.
 * @param input_files Vector of input file paths to be processed for this shard.
 * @param num_threads Number of threads to use for parallel processing of input files.
 * @param tmp_dir Temporary directory path for intermediate files.
 */
void process_shard(
    int shard_idx,
    const std::string& dict_centroids,
    const std::vector<std::string>& input_files,
    int num_threads,
    std::string tmp_dir)
{

    std::unordered_map<std::string, std::string> prot_to_centroid_dict;
    std::vector<uint64_t> line_positions;

    std::cout << "Loading dictionary for shard " << shard_idx << std::endl;
    std::string positions_file = tmp_dir + "hash_positions_" + std::to_string(shard_idx) + ".bin";
    std::ifstream pos_file(positions_file, std::ios::binary | std::ios::ate);
    if (!pos_file) {
        std::cerr << "ERROR: Cannot open " << positions_file << std::endl;
        return;
    }
    std::streamsize file_size = pos_file.tellg();
    pos_file.seekg(0, std::ios::beg);
    size_t total_positions = file_size / 8;

    size_t count = 0;
    auto start_time = std::chrono::steady_clock::now();
    while (pos_file) {
        uint64_t pos;
        pos_file.read(reinterpret_cast<char*>(&pos), sizeof(uint64_t));
        if (!pos_file) break;
        line_positions.push_back(pos);
        ++count;
        if (count % 10000000 == 0) {
            auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            double lines_per_sec = count / (elapsed > 0 ? elapsed : 1);
            size_t remaining = total_positions > count ? total_positions - count : 0;
            double eta_sec = lines_per_sec > 0 ? remaining / lines_per_sec : 0;
            int eta_min = static_cast<int>(eta_sec / 60);
            std::cout << "Shard " << shard_idx << ": loaded " << count << "/" << total_positions
                      << " positions, elapsed: " << (elapsed / 60) << " min, ETA: " << eta_min << " min" << std::endl;
        }
    }

    std::cout << "Loaded the positions of the lines, now loading the actual dict" << std::endl;

    std::ifstream dict_file(dict_centroids);
    if (!dict_file) {
        std::cerr << "ERROR: Cannot open " << dict_centroids << std::endl;
        return;
    }
    total_positions = line_positions.size();
    start_time = std::chrono::steady_clock::now();
    for (size_t idx = 0; idx < line_positions.size(); ++idx) {
        dict_file.clear();
        dict_file.seekg(line_positions[idx]);
        std::string line;
        std::getline(dict_file, line);
        if (line.empty()) continue;
        size_t tab_pos = line.find(' '); //now it's space-separated, not tab_separated
        if (tab_pos == std::string::npos) continue;
        std::string centroid = line.substr(0, tab_pos);
        std::string protein = line.substr(tab_pos + 1);
        prot_to_centroid_dict[protein] = centroid;
        if ((idx + 1) % 10000000 == 0) {
            auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
            double lines_per_sec = (idx + 1) / (elapsed > 0 ? elapsed : 1);
            size_t remaining = total_positions > (idx + 1) ? total_positions - (idx + 1) : 0;
            double eta_sec = lines_per_sec > 0 ? remaining / lines_per_sec : 0;
            int eta_min = static_cast<int>(eta_sec / 60);
            std::cout << "Shard " << shard_idx << ": loaded " << (idx + 1) << "/" << total_positions
                      << " dict entries, elapsed: " << (elapsed / 60) << " min, ETA: " << eta_min << " min" << std::endl;
        }
    }

    std::cout << "Loaded the dictionary" << std::endl;

    // Process all input files in parallel
    std::vector<std::future<void>> futures;
    size_t file_count = input_files.size();
    std::atomic<size_t> next_file(0);

    auto worker = [&]() {
        while (true) {
            size_t idx = next_file.fetch_add(1); //increment in a thread-safe way
            if (idx >= file_count) break;
            process_input_file_for_shard(
                input_files[idx],
                shard_idx,
                prot_to_centroid_dict,
                tmp_dir
            );
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }
    for (auto& t : threads) t.join();

    std::cout << "Finished indexing for shard " << shard_idx << std::endl;
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: create_all_prots_database <shard_idx>" << std::endl;
        return 1;
    }

    int shard_idx;
    try {
        shard_idx = std::stoi(argv[1]);
    } catch (const std::exception&) {
        std::cerr << "ERROR: shard_idx must be an integer" << std::endl;
        return 1;
    }

    // const int num_shards = 100;
    // int num_threads = std::max(1u, std::thread::hardware_concurrency() * 2);
    // const std::string input_folder = "/pasteur/appa/scratch/rchikhi/logan_cluster/orfs";
    // const std::string dict_centroids = "/pasteur/appa/scratch/rfaure/human-complete.tsv";
    // const std::string output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins_human/";

    const int num_shards = 100;
    int num_threads = 2; //over-use the CPUs to exploit the fact that thread wait for I/O on disk
    const std::string input_folder = "/pasteur/appa/scratch/rfaure/orfs";
    const std::string dict_centroids = "/pasteur/appa/scratch/rfaure/nonhuman-complete_tsv_sorted/nonhuman-complete.tsv";
    const std::string output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins3/";
    std::string pattern = "nonhuman";

    //// Test configuration for all_prots_test
    // const int num_shards = 2;
    // int num_threads = 10;
    // const std::string file_centroids = "/pasteur/appa/scratch/rfaure/all_prots_test/centroids.fa";
    // const std::string input_folder = "/pasteur/appa/scratch/rfaure/all_prots_test";
    // const std::string dict_centroids = "/pasteur/appa/scratch/rfaure/all_prots_test/centroid.tsv";
    // const std::string output_dir = "/pasteur/appa/scratch/rfaure/all_prots_test/proteins/";

    const std::string tmp_dir = output_dir + "tmp/";

    std::vector<std::string> input_files;
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.rfind(pattern, 0) == 0 && filename.size() >= 4 && filename.substr(filename.size() - 4) == ".zst") {
                input_files.push_back(entry.path().string());
            }
        }
    }

    std::cout << "Input files for shard " << shard_idx << ":" << std::endl;
    for (const auto& file : input_files) {
        std::cout << "  " << file << std::endl;
    }

    process_shard(shard_idx, dict_centroids, input_files, num_threads, tmp_dir);
  
    cout << "FINISHED SHARD " << shard_idx << endl;

}
