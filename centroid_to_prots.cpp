#include <iostream>
#include <string>

#include <fstream>
#include <zstd.h>
#include <vector>
#include <sstream>


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

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_files> <centroid_id>" << std::endl;
        return 1;
    }

    std::string centroid_id = argv[2];
    auto centroid_hash = simple_string_hash(centroid_id)%10000;

    try {
        std::string filename = std::string(argv[1])+"/centroid_"+std::to_string(centroid_hash)+"/"+std::to_string((simple_string_hash(centroid_id)/10000)%1000)+".fa.zst";
        std::string cmd = "zstdcat " + filename;
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            throw std::runtime_error("Failed to run zstdcat command");
        }
        std::string file_content;
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            file_content += buffer;
        }
        int ret_code = pclose(pipe);
        if (ret_code != 0) {
            throw std::runtime_error("zstdcat command failed");
        }

        std::istringstream iss(file_content);
        std::string line, header, sequence;
        while (std::getline(iss, line)) {
            if (!line.empty() && line[0] == '>') {
                if (!header.empty() && !sequence.empty()) {
                    // Check if header starts with the centroid_id
                    if (header.substr(1, centroid_id.size()) == centroid_id) {
                        std::cout << header << '\n' << sequence << '\n';
                    }
                }
                header = line;
                sequence.clear();
            } else {
                sequence += line;
            }
        }
        // Check last sequence
        if (!header.empty() && !sequence.empty()) {
            // Check if header starts with the centroid_id
            if (header.compare(1, centroid_id.size(), centroid_id) == 0) {
                std::cout << header << '\n' << sequence << '\n';
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}