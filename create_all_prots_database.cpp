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

// Global file locks for thread safety
std::map<std::string, std::mutex> file_locks;

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

// Helper to get or create a lock for a file
std::mutex& get_file_lock(const std::string& filename) {
    static std::mutex global_mutex;
    std::lock_guard<std::mutex> guard(global_mutex);
    return file_locks[filename];
}

// prot_to_centroid_dict: unordered_map<string, string>
void process_input_file_for_shard(
    const std::string& input_file,
    int num_centroids_shards,
    int shard_idx,
    const std::unordered_map<std::string, std::string>& prot_to_centroid_dict,
    std::string tmp_dir)
{
    auto start_time = std::chrono::steady_clock::now();
    std::string base = fs::path(input_file).filename().string();
    std::string shard_file = tmp_dir + base + "_shard_" + std::to_string(shard_idx) + ".fa";
    if (!fs::exists(shard_file)) {
        return;
    }

    //Let's find, for each file, what was the last dispatched protein of the file in the previous run
    // Open the shard_file and get the last protein ID (header) that is in prot_to_centroid_dict
    std::string last_dispatched_protein_id;
    {
        std::ifstream fasta_handle(shard_file);
        std::string line;
        while (std::getline(fasta_handle, line)) {
            if (line.empty() || line[0] != '>') continue;
            std::string protein_id = line.substr(1);
            size_t space_pos = protein_id.find(' ');
            if (space_pos != std::string::npos){
                protein_id = protein_id.substr(0, space_pos);
            }
            if (prot_to_centroid_dict.find(protein_id) != prot_to_centroid_dict.end()) {
                last_dispatched_protein_id = protein_id;
            }
            // Skip sequence line
            std::getline(fasta_handle, line);
        }
    }

    // Check if the last dispatched protein has already been dispatched
    if (!last_dispatched_protein_id.empty()) {
        auto it = prot_to_centroid_dict.find(last_dispatched_protein_id);
        if (it != prot_to_centroid_dict.end()) {
            const std::string& centroid = it->second;
            int centroid_hash = simple_string_hash(centroid) % num_centroids_shards;
            std::string output_file = tmp_dir + "centroid_" + std::to_string(centroid_hash) + ".fa";
            std::ifstream out_handle(output_file);
            if (out_handle) {
                std::string line;
                std::string search_str = ">" + centroid + " " + last_dispatched_protein_id;
                while (std::getline(out_handle, line)) {
                    if (line.rfind(search_str, 0) == 0) {
                        std::cout << "Last dispatched protein " << last_dispatched_protein_id << " already dispatched, skipping file " << shard_file << std::endl;
                        return;
                    }
                }
            }
        }
    }

    int number_of_missed_proteins = 0;
    int number_of_dispatched_proteins = 0;

    std::ifstream fasta_handle(shard_file);
    if (!fasta_handle) return;

    std::string header_line;
    while (std::getline(fasta_handle, header_line)) {
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
            // Skip sequence line
            std::string skip_seq;
            std::getline(fasta_handle, skip_seq);
            number_of_missed_proteins += 1;
            continue;
        }
        number_of_dispatched_proteins += 1;
        const std::string& centroid = it->second;
        // size_t centroid_hash = std::hash<std::string>{}(centroid) % num_centroids_shards;
        int centroid_hash = simple_string_hash(centroid) % num_centroids_shards;
        std::string output_file = tmp_dir + "centroid_" + std::to_string(centroid_hash) + ".fa";

        // Build new description
        auto new_desc = centroid + " " + header_line.substr(1);

        std::string sequence;
        std::getline(fasta_handle, sequence);

        // Lock file for thread safety
        std::mutex& lock = get_file_lock(output_file);
        {
            std::lock_guard<std::mutex> guard(lock);
            std::ofstream out_handle(output_file, std::ios::app);
            out_handle << ">" << new_desc << "\n" << sequence << "\n";
        }
    }

    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "In shard " << shard_idx << ", just dispatched " << shard_file
              << " in " << elapsed << " seconds. In total, dispatched " << number_of_dispatched_proteins << " and did not find " << number_of_missed_proteins << " proteins." << std::endl;
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
 * @param num_centroids_shards Total number of centroid shards.
 * @param tmp_dir Temporary directory path for intermediate files.
 */
void process_shard(
    int shard_idx,
    const std::string& dict_centroids,
    const std::vector<std::string>& input_files,
    int num_threads,
    int num_centroids_shards,
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
        size_t tab_pos = line.find('\t');
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
                num_centroids_shards,
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
}

void sort_and_split_centroid_file(std::string shard_idx, std::string output_dir){
    string file_to_split = output_dir + "tmp/centroid_"+shard_idx+".fa";

    std::string sort_cmd = "awk '{cur=$0; if(length(cur)<3){prev=\"\"; next}; if(prev!=\"\")print prev; prev=cur} END{if(prev!=\"\")print prev}' " + file_to_split +
        " | grep -A1 --no-group-separator '^>' | paste - - | sort -k1,1 | awk '!seen[$0]++' | tr '\\t' '\\n' > " + file_to_split + ".sorted";
    int sort_ret = std::system(sort_cmd.c_str());
    if (sort_ret != 0) {
        std::cerr << "ERROR: Failed to sort " << file_to_split << std::endl;
        return;
    }
    file_to_split = file_to_split + ".sorted";

    cout << " splitting " << file_to_split << endl;
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
    // int num_threads = 24;
    // int num_centroids_shards = 10000;
    // const std::string input_folder = "/pasteur/appa/scratch/rchikhi/logan_cluster/orfs/";
    // const std::string dict_centroids = "/pasteur/appa/scratch/rfaure/human-complete.tsv";
    // const std::string output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins_human/";

    const int num_shards = 100;
    int num_threads = 24;
    int num_centroids_shards = 10000;
    const std::string input_folder = "/pasteur/appa/scratch/rchikhi/logan_cluster/orfs/";
    const std::string dict_centroids = "/pasteur/appa/scratch/rfaure/nonhuman-complete.tsv";
    const std::string output_dir = "/pasteur/appa/scratch/rfaure/all_prots/proteins3/";

    //// Test configuration for all_prots_test
    // const int num_shards = 2;
    // int num_threads = 10;
    // int num_centroids_shards = 2;
    // const std::string file_centroids = "/pasteur/appa/scratch/rfaure/all_prots_test/centroids.fa";
    // const std::string input_folder = "/pasteur/appa/scratch/rfaure/all_prots_test";
    // const std::string dict_centroids = "/pasteur/appa/scratch/rfaure/all_prots_test/centroid.tsv";
    // const std::string output_dir = "/pasteur/appa/scratch/rfaure/all_prots_test/proteins/";

    const std::string tmp_dir = output_dir + "tmp/";

    std::vector<std::string> input_files;
    for (const auto& entry : fs::directory_iterator(input_folder)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            if (filename.rfind("nonhuman", 0) == 0 && filename.size() >= 4 && filename.substr(filename.size() - 4) == ".zst") {
                input_files.push_back(entry.path().string().substr(0, entry.path().string().size() - 4));
            }
        }
    }

    std::cout << "Input files for shard " << shard_idx << ":" << std::endl;
    for (const auto& file : input_files) {
        std::cout << "  " << file << std::endl;
    }

    process_shard(shard_idx, dict_centroids, input_files, num_threads, num_centroids_shards, tmp_dir);

    // for (int i = 100*shard_idx ; i<= 100*(shard_idx+1) ; i++){
    //     sort_and_split_centroid_file(std::to_string(i), output_dir);
    // }
  
    cout << "FINISHED SHARD " << shard_idx << endl;

}