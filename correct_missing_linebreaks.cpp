//in the all-proteins file, sometimes there are headers at the end of sequences lines. The problem is not solved yet...

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using std::vector;
using std::string;
using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    std::string new_file = std::string(argv[1]) + ".tmp";
    std::string cmd = "zstd -d -c " + std::string(argv[1]) + " > " + new_file;
    cout << "running command " << cmd << endl;
    system(cmd.c_str());

    std::ifstream infile(new_file);

    std::string new_file2 = std::string(argv[1]) + ".tmp2";
    std::ofstream outfile(new_file2);

    std::vector<std::pair<string,string>> unclassified_records;
    bool next_is_unclassified = false;
    string unclassified_name;
    std::string line;
    while (std::getline(infile, line)) {
        size_t pos = line.find('>');
        if (next_is_unclassified){
            next_is_unclassified = false;
            unclassified_records.push_back({unclassified_name, line});
        }
        else if (pos != std::string::npos && pos != 0) { //we found a > in the middle of the line !
            outfile << line.substr(0, pos) << "\n";
            unclassified_name = line.substr(pos); // the rest of the line, including the >
            next_is_unclassified = true;
        } else {
            outfile << line << "\n";
        }
    }

    outfile.close();
    infile.close();

    for (const auto& record : unclassified_records) {
        std::cout << record.first << "\n" << record.second << "\n";
    }

    return 0;
}