#ifndef GRAPH_GENERATION_HPP
#define GRAPH_GENERATION_HPP

#include <vector>

/**
 * @brief Generates a random hypergraph represented as hyperedge-to-vertex mapping
 * 
 * @param n Number of hyperedges to generate
 * @param m Maximum number of vertices per hyperedge
 * @param r1 Minimum vertex ID value
 * @param r2 Maximum vertex ID value
 * @return std::vector<std::vector<int>> 2D vector where each row represents a hyperedge
 *         and contains the vertex IDs that belong to that hyperedge
 */
std::vector<std::vector<int>> hyperedge2vertex(int n, int m, int r1, int r2);

/**
 * @brief Creates the inverse mapping from vertex-to-hyperedge
 * 
 * @param hyperedgeToVertex The hyperedge-to-vertex mapping
 * @return std::vector<std::vector<int>> 2D vector where each row represents a vertex
 *         and contains the hyperedge IDs that contain that vertex
 */
std::vector<std::vector<int>> vertex2hyperedge(const std::vector<std::vector<int>>& hyperedgeToVertex);

/**
 * @brief Prints a 2D vector in matrix form
 * 
 * @param vec2d The 2D vector to print
 */
void print2DVector(const std::vector<std::vector<int>>& vec2d);

#endif // GRAPH_GENERATION_HPP
