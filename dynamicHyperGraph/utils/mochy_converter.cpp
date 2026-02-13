#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " <input_space_separated.txt> <output_comma_separated.txt>\n";
    std::cerr << "Example: " << program << " input/Coauth.txt MoCHy-Master/data/Coauth.csv\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 3) {
        printUsage(argv[0]);
        return 1;
    }

    const std::string inputPath = argv[1];
    const std::string outputPath = argv[2];

    std::ifstream in(inputPath);
    if (!in.is_open()) {
        std::cerr << "Error: failed to open input file: " << inputPath << "\n";
        return 1;
    }

    std::ofstream out(outputPath);
    if (!out.is_open()) {
        std::cerr << "Error: failed to open output file: " << outputPath << "\n";
        return 1;
    }

    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string tok;
        while (iss >> tok) {
            tokens.push_back(tok);
        }
        if (tokens.empty()) {
            continue;
        }

        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i > 0) out << ",";
            out << tokens[i];
        }
        out << "\n";
    }

    std::cout << "Converted hyperedge list to MoCHy format:\n";
    std::cout << "  Input : " << inputPath << "\n";
    std::cout << "  Output: " << outputPath << "\n";
    return 0;
}
