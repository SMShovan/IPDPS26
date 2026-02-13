#include "../include/utils.hpp"
#include "../include/graphGeneration.hpp"
#include "../include/printUtils.hpp"
#include <iostream>
#include <cstdlib>
#include <set>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <limits>

namespace {

const char* kInputDirectory = "input";
const char* kDefaultGeneratedFile = "generated_hypergraph.txt";

bool ensureInputDirectory() {
    std::error_code ec;
    std::filesystem::create_directories(kInputDirectory, ec);
    if (ec) {
        std::cerr << "Error: failed to create input directory '" << kInputDirectory
                  << "': " << ec.message() << std::endl;
        return false;
    }
    return true;
}

bool parsePositiveInt(const std::string& token, int& value) {
    try {
        size_t consumed = 0;
        int parsed = std::stoi(token, &consumed);
        if (consumed != token.size() || parsed <= 0) return false;
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseNonNegativeDouble(const std::string& token, double& value) {
    try {
        size_t consumed = 0;
        double parsed = std::stod(token, &consumed);
        if (consumed != token.size() || parsed < 0.0) return false;
        value = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

void printUsage(const char* programName) {
    std::cerr << "Usage (generate mode): " << programName
              << " <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id> <payload_capacity> [alignment=4] "
              << "[-totalChanges N] [-insertionPercentage P] [-deletionPercentage P] [--save-generated[=FILE]]"
              << std::endl;
    std::cerr << "Usage (input mode):    " << programName
              << " <payload_capacity> [alignment=4] --input=FILE "
              << "[-totalChanges N] [-insertionPercentage P] [-deletionPercentage P]" << std::endl;
}

std::string buildInputPath(const std::string& fileName) {
    return std::string(kInputDirectory) + "/" + fileName;
}

bool saveHypergraphToFile(const std::vector<std::vector<int>>& hyperedgeToVertex, const std::string& fileName) {
    if (!ensureInputDirectory()) return false;
    const std::string outputPath = buildInputPath(fileName.empty() ? kDefaultGeneratedFile : fileName);

    std::ofstream out(outputPath);
    if (!out.is_open()) {
        std::cerr << "Error: failed to open output file '" << outputPath << "'" << std::endl;
        return false;
    }

    for (const auto& row : hyperedgeToVertex) {
        for (size_t i = 0; i < row.size(); ++i) {
            if (i > 0) out << ' ';
            out << row[i];
        }
        out << '\n';
    }

    std::cout << "Saved generated hypergraph to " << outputPath << std::endl;
    return true;
}

bool loadHypergraphFromFile(const std::string& fileName, std::vector<std::vector<int>>& hyperedgeToVertex) {
    if (!ensureInputDirectory()) return false;
    const std::string inputPath = buildInputPath(fileName);

    std::ifstream in(inputPath);
    if (!in.is_open()) {
        std::cerr << "Error: input file not found: " << inputPath << std::endl;
        return false;
    }

    hyperedgeToVertex.clear();
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line)) {
        ++lineNo;
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::istringstream iss(line);
        std::vector<int> vertices;
        int vertexId = 0;
        while (iss >> vertexId) {
            if (vertexId <= 0) {
                std::cerr << "Error: non-positive vertex id at line " << lineNo << std::endl;
                return false;
            }
            vertices.push_back(vertexId);
        }

        if (vertices.empty()) {
            continue;
        }
        hyperedgeToVertex.push_back(std::move(vertices));
    }

    if (hyperedgeToVertex.empty()) {
        std::cerr << "Error: input file contains no hyperedges: " << inputPath << std::endl;
        return false;
    }

    std::cout << "Loaded hypergraph from " << inputPath << std::endl;
    return true;
}

} // namespace

// Function to parse and validate command line arguments
bool parseCommandLineArgs(int argc, char* argv[], HypergraphParams& params) {
    if (argc < 3) {
        printUsage(argv[0]);
        std::cerr << "Example (generate): " << argv[0] << " 8 5 1 100 4096 8 --save-generated=run1.txt" << std::endl;
        std::cerr << "Example (input):    " << argv[0] << " 4096 8 --input=run1.txt" << std::endl;
        return false;
    }

    params.numHyperedges = 0;
    params.maxVerticesPerHyperedge = 0;
    params.minVertexId = 0;
    params.maxVertexId = 0;
    params.payloadCapacity = 0;
    params.alignment = 4;
    params.inputFileName.clear();
    params.saveGenerated = false;
    params.saveGeneratedFileName.clear();
    params.totalChanges = 5;
    params.insertionPercentage = 50.0;
    params.deletionPercentage = 50.0;

    std::vector<std::string> positionalArgs;
    positionalArgs.reserve(static_cast<size_t>(argc));
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--input=", 0) == 0) {
            params.inputFileName = arg.substr(std::string("--input=").size());
            if (params.inputFileName.empty()) {
                std::cerr << "Error: --input requires a file name under input/" << std::endl;
                return false;
            }
            continue;
        }

        if (arg == "--save-generated") {
            params.saveGenerated = true;
            continue;
        }

        if (arg.rfind("--save-generated=", 0) == 0) {
            params.saveGenerated = true;
            params.saveGeneratedFileName = arg.substr(std::string("--save-generated=").size());
            if (params.saveGeneratedFileName.empty()) {
                std::cerr << "Error: --save-generated requires a non-empty file name when using '=' form" << std::endl;
                return false;
            }
            continue;
        }

        if (arg == "-totalChanges" || arg == "--totalChanges") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " expects a value" << std::endl;
                return false;
            }
            if (!parsePositiveInt(argv[++i], params.totalChanges)) {
                std::cerr << "Error: totalChanges must be a positive integer" << std::endl;
                return false;
            }
            continue;
        }
        if (arg.rfind("-totalChanges=", 0) == 0 || arg.rfind("--totalChanges=", 0) == 0) {
            std::string value = arg.substr(arg.find('=') + 1);
            if (!parsePositiveInt(value, params.totalChanges)) {
                std::cerr << "Error: totalChanges must be a positive integer" << std::endl;
                return false;
            }
            continue;
        }

        if (arg == "-insertionPercentage" || arg == "--insertionPercentage") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " expects a value" << std::endl;
                return false;
            }
            if (!parseNonNegativeDouble(argv[++i], params.insertionPercentage)) {
                std::cerr << "Error: insertionPercentage must be a non-negative number" << std::endl;
                return false;
            }
            continue;
        }
        if (arg.rfind("-insertionPercentage=", 0) == 0 || arg.rfind("--insertionPercentage=", 0) == 0) {
            std::string value = arg.substr(arg.find('=') + 1);
            if (!parseNonNegativeDouble(value, params.insertionPercentage)) {
                std::cerr << "Error: insertionPercentage must be a non-negative number" << std::endl;
                return false;
            }
            continue;
        }

        if (arg == "-deletionPercentage" || arg == "--deletionPercentage") {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << arg << " expects a value" << std::endl;
                return false;
            }
            if (!parseNonNegativeDouble(argv[++i], params.deletionPercentage)) {
                std::cerr << "Error: deletionPercentage must be a non-negative number" << std::endl;
                return false;
            }
            continue;
        }
        if (arg.rfind("-deletionPercentage=", 0) == 0 || arg.rfind("--deletionPercentage=", 0) == 0) {
            std::string value = arg.substr(arg.find('=') + 1);
            if (!parseNonNegativeDouble(value, params.deletionPercentage)) {
                std::cerr << "Error: deletionPercentage must be a non-negative number" << std::endl;
                return false;
            }
            continue;
        }
        positionalArgs.push_back(arg);
    }

    if (!params.inputFileName.empty()) {
        // Input mode: only payload capacity and optional alignment are required.
        if (positionalArgs.size() < 1 || positionalArgs.size() > 2) {
            std::cerr << "Error: input mode expects <payload_capacity> [alignment=4] plus --input=FILE" << std::endl;
            printUsage(argv[0]);
            return false;
        }

        if (!parsePositiveInt(positionalArgs[0], params.payloadCapacity)) {
            std::cerr << "Error: payload_capacity must be a positive integer" << std::endl;
            return false;
        }
        if (positionalArgs.size() == 2 && !parsePositiveInt(positionalArgs[1], params.alignment)) {
            std::cerr << "Error: alignment must be a positive integer" << std::endl;
            return false;
        }
        if (params.saveGenerated) {
            std::cout << "Warning: --save-generated is ignored when --input is provided." << std::endl;
            params.saveGenerated = false;
            params.saveGeneratedFileName.clear();
        }
    } else {
        // Generate mode: full generation parameters are required.
        if (positionalArgs.size() < 5 || positionalArgs.size() > 6) {
            std::cerr << "Error: generate mode expects <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id> <payload_capacity> [alignment=4]" << std::endl;
            printUsage(argv[0]);
            return false;
        }

        if (!parsePositiveInt(positionalArgs[0], params.numHyperedges) ||
            !parsePositiveInt(positionalArgs[1], params.maxVerticesPerHyperedge) ||
            !parsePositiveInt(positionalArgs[2], params.minVertexId) ||
            !parsePositiveInt(positionalArgs[3], params.maxVertexId) ||
            !parsePositiveInt(positionalArgs[4], params.payloadCapacity)) {
            std::cerr << "Error: all generate-mode numeric arguments must be positive integers" << std::endl;
            return false;
        }
        if (positionalArgs.size() == 6 && !parsePositiveInt(positionalArgs[5], params.alignment)) {
            std::cerr << "Error: alignment must be a positive integer" << std::endl;
            return false;
        }

        if (params.minVertexId >= params.maxVertexId) {
            std::cerr << "Error: min_vertex_id must be less than max_vertex_id" << std::endl;
            return false;
        }
    }

    if (params.alignment <= 0 || params.payloadCapacity <= 0) {
        std::cerr << "Error: payload_capacity and alignment must be positive integers" << std::endl;
        return false;
    }
    if (params.totalChanges <= 0) {
        std::cerr << "Error: totalChanges must be a positive integer" << std::endl;
        return false;
    }
    const double pctSum = params.insertionPercentage + params.deletionPercentage;
    if (params.insertionPercentage < 0.0 || params.deletionPercentage < 0.0 || pctSum <= 0.0) {
        std::cerr << "Error: insertionPercentage and deletionPercentage must be non-negative, and their sum must be > 0" << std::endl;
        return false;
    }

    return true;
}

// Function to print hypergraph parameters
void printHypergraphParams(const HypergraphParams& params) {
    std::cout << "Configured run parameters:" << std::endl;
    std::cout << "  - Payload capacity: " << params.payloadCapacity << std::endl;
    std::cout << "  - Alignment: " << params.alignment << std::endl;
    if (!params.inputFileName.empty()) {
        std::cout << "  - Mode: input" << std::endl;
        std::cout << "  - Input file: input/" << params.inputFileName << std::endl;
        std::cout << "  - Graph generation parameters: inferred from input file" << std::endl;
    } else {
        std::cout << "  - Mode: generate" << std::endl;
        std::cout << "  - " << params.numHyperedges << " hyperedges" << std::endl;
        std::cout << "  - Up to " << params.maxVerticesPerHyperedge << " vertices per hyperedge" << std::endl;
        std::cout << "  - Vertex IDs in range [" << params.minVertexId << ", " << params.maxVertexId << "]" << std::endl;
    }
    if (params.saveGenerated) {
        const std::string target = params.saveGeneratedFileName.empty() ? kDefaultGeneratedFile : params.saveGeneratedFileName;
        std::cout << "  - Save generated hypergraph: input/" << target << std::endl;
    }
    std::cout << "  - Update batch totalChanges: " << params.totalChanges << std::endl;
    std::cout << "  - Insertion percentage: " << params.insertionPercentage << std::endl;
    std::cout << "  - Deletion percentage: " << params.deletionPercentage << std::endl;
    std::cout << std::endl;
}

// Function to generate hypergraph mappings
std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> generateHypergraph(const HypergraphParams& params) {
    // Generate hypergraph data structures
    std::vector<std::vector<int>> hyperedgeToVertex = hyperedge2vertex(params.numHyperedges, params.maxVerticesPerHyperedge, params.minVertexId, params.maxVertexId);
    std::vector<std::vector<int>> vertexToHyperedge = vertex2hyperedge(hyperedgeToVertex);
    
    // Display the generated mappings
    std::cout << "Hyperedge to vertex" << std::endl;
    print2DVector(hyperedgeToVertex);
    std::cout << "Vertex to hyperedge" << std::endl;
    print2DVector(vertexToHyperedge);
    
    return {hyperedgeToVertex, vertexToHyperedge};
}

std::pair<std::vector<std::vector<int>>, std::vector<std::vector<int>>> loadOrGenerateHypergraph(HypergraphParams& params) {
    std::vector<std::vector<int>> hyperedgeToVertex;
    if (!params.inputFileName.empty()) {
        if (!loadHypergraphFromFile(params.inputFileName, hyperedgeToVertex)) {
            return {};
        }
        params.numHyperedges = static_cast<int>(hyperedgeToVertex.size());
        params.maxVerticesPerHyperedge = 0;
        params.minVertexId = std::numeric_limits<int>::max();
        params.maxVertexId = 0;
        for (const auto& row : hyperedgeToVertex) {
            params.maxVerticesPerHyperedge = std::max(params.maxVerticesPerHyperedge, static_cast<int>(row.size()));
            for (int vertexId : row) {
                params.minVertexId = std::min(params.minVertexId, vertexId);
                params.maxVertexId = std::max(params.maxVertexId, vertexId);
            }
        }
        if (params.minVertexId == std::numeric_limits<int>::max()) {
            params.minVertexId = 0;
        }
        std::cout << "Inferred from input: "
                  << params.numHyperedges << " hyperedges, max vertices per hyperedge "
                  << params.maxVerticesPerHyperedge << ", vertex range ["
                  << params.minVertexId << ", " << params.maxVertexId << "]"
                  << std::endl;
        std::cout << "Hyperedge to vertex" << std::endl;
        print2DVector(hyperedgeToVertex);
    } else {
        hyperedgeToVertex = hyperedge2vertex(params.numHyperedges, params.maxVerticesPerHyperedge, params.minVertexId, params.maxVertexId);
        std::cout << "Hyperedge to vertex" << std::endl;
        print2DVector(hyperedgeToVertex);

        if (params.saveGenerated) {
            const std::string outputName = params.saveGeneratedFileName.empty() ? kDefaultGeneratedFile : params.saveGeneratedFileName;
            if (!saveHypergraphToFile(hyperedgeToVertex, outputName)) {
                return {};
            }
        }
    }

    std::vector<std::vector<int>> vertexToHyperedge = vertex2hyperedge(hyperedgeToVertex);
    std::cout << "Vertex to hyperedge" << std::endl;
    print2DVector(vertexToHyperedge);

    return {hyperedgeToVertex, vertexToHyperedge};
}

// Function to generate hyperedge-to-hyperedge adjacency
std::vector<std::vector<int>> hyperedgeAdjacency(const std::vector<std::vector<int>>& vertexToHyperedge, const std::vector<std::vector<int>>& hyperedgeToVertex) {
    int nHyperedges = hyperedgeToVertex.size();
    
    // Resultant adjacency matrix for hyperedges (store 1-based hyperedge IDs)
    std::vector<std::vector<int>> hyperedgeAdjacencyMatrix(nHyperedges);

    // Iterate through each hyperedge (0-indexed)
    for (int hyperedge = 0; hyperedge < nHyperedges; ++hyperedge) {
        std::set<int> adjacentHyperedges;

        // Get the vertices connected by this hyperedge
        const std::vector<int>& vertices = hyperedgeToVertex[hyperedge];

        // For each vertex, find other hyperedges connected to it
        for (int vertex : vertices) {
            for (int otherHyperedge : vertexToHyperedge[vertex]) {
                // vertexToHyperedge stores 1-based hyperedge IDs; avoid self-loop by comparing to (hyperedge + 1)
                if (otherHyperedge != hyperedge + 1) {
                    adjacentHyperedges.insert(otherHyperedge); // Ensure no duplicates
                }
            }
        }

        // Convert set to vector and store in adjacency matrix
        hyperedgeAdjacencyMatrix[hyperedge] = std::vector<int>(adjacentHyperedges.begin(), adjacentHyperedges.end());
    }

    return hyperedgeAdjacencyMatrix;
}

// Function to flatten 2D vector and print debug info
std::pair<std::vector<int>, std::vector<int>> flatten(const std::vector<std::vector<int>>& vec2d, const std::string& name) {
    auto flattened = flatten2DVector(vec2d);
    
    // Print the flattened vectors
    printVector(flattened.first, "Flattened Values (" + name + ")");
    printVector(flattened.second, "Flattened Indices (" + name + ")");
    
    return flattened;
}

// Function to prepare CBST data
std::pair<int*, int*> prepareCBSTData(const std::vector<int>& flatIndices) {
    int* h_values = const_cast<int*>(flatIndices.data());
    int* h_indices = new int[flatIndices.size()];
    for (size_t i = 0; i < flatIndices.size(); ++i) {
        h_indices[i] = i + 1;
    }
    return {h_values, h_indices};
}
