#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <iostream>
#include <filesystem>

#include "libparsers.hpp"
#include "dcgnn_matrix.hpp"

int parse_dimacs_to_dcgnn_vcg(const char base_dir[256], const char file_name[256], const unsigned label)
{
    using namespace std::string_literals;
    using namespace master_thesis;
    namespace fs = std::filesystem;
    
    const auto file_path = std::string{base_dir} + "/"s + std::string{file_name};
    if (!fs::exists(file_path))
    {
        std::cout << "The file does not exist: " << file_path << std::endl;
        return -1;
    }

    std::ifstream input{file_path};
    if (!input.is_open())
    {
        std::cout << "Error opening file: " << file_path << std::endl;
        return -1;
    }

    std::cout << "Parsing file: " << file_path << std::endl;

    dcgnn_matrix<unsigned> vcg{label};
    unsigned num_of_vars;
    unsigned num_of_clauses;

    std::string line;
    unsigned clause_idx = 0u;
    while (std::getline(input, line))
    {
        if (line.empty() || line[0] == 'c') 
        {
            continue;
        }
        if (line[0] == 'p')
        {
            assert(!clause_idx);
            std::istringstream dimacs_header_line{line};
            std::string problem_data;
            dimacs_header_line >> problem_data
                               >> problem_data
                               >> num_of_vars
                               >> num_of_clauses;
            vcg.resize(2*num_of_vars + num_of_clauses);
            continue;
        }
        
        std::istringstream clause{line};
        long var_node;
        while (clause >> var_node)
        {
            if (!var_node) 
            {
                break;
            }
            // Calculate the var node index
            unsigned var_idx = (var_node > 0) 
                ? (static_cast<unsigned>(var_node) - 1u + num_of_clauses) 
                : (static_cast<unsigned>(-var_node) - 1u + num_of_clauses + num_of_vars);

            // Connect the clause with the node
            vcg.add_a_neighbour(clause_idx, var_idx);
            // Connect the node with the clause
            vcg.add_a_neighbour(var_idx, clause_idx);
            // Set the tag for the var node
            vcg.set_tag_for_node(var_idx, (var_node > 0) ? "1" : "-1");
        }

        // Set the tag for the clause node
        vcg.set_tag_for_node(clause_idx, "0");
        // Move on to the next clause
        ++clause_idx;
    }

    const auto output_dir = std::string{base_dir} + "/parsed/"s;
    const auto output_file = std::string{file_name} + ".txt"s;
    vcg.save_to_file(output_dir, output_file);
    return 0;
}