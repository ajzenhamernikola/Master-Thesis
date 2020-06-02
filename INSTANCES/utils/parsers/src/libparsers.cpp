#include <algorithm>
#include <fstream>
#include <string>
#include <sstream>
#include <cassert>
#include <iostream>
#include <filesystem>

#include "libparsers.hpp"
#include "dcgnn_graph.hpp"
#include "edgelist_graph.hpp"

int parse_dimacs_to_dcgnn_vcg(const char base_dir[256], const char file_name[256], const unsigned label)
{
    using namespace std::string_literals;
    using namespace MasterThesis;
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

    auto vcg = std::make_shared<DcgnnGraph>(label);
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
            vcg->Resize(2*num_of_vars + num_of_clauses);
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
            vcg->AddANeighbour(clause_idx, var_idx);
            // Connect the node with the clause
            vcg->AddANeighbour(var_idx, clause_idx);
        }

        // Set the tag for the clause node
        vcg->SetTagForNode(clause_idx, "0");
        // Move on to the next clause
        ++clause_idx;
    }

    // Set the tags for all var nodes
    for (auto var_node = 1u; var_node <= num_of_vars; ++var_node)
    {
        auto positive_var_idx = static_cast<unsigned>(var_node) - 1u + num_of_clauses;
        vcg->SetTagForNode(positive_var_idx, "1");

        auto negative_var_idx = static_cast<unsigned>(var_node) - 1u + num_of_clauses + num_of_vars;
        vcg->SetTagForNode(negative_var_idx, "-1");
    }

    const auto output_dir = std::string{base_dir} + "/parsed/"s;
    const auto output_file = std::string{file_name} + ".dcgnn.txt"s;
    vcg->SaveToFile(output_dir, output_file);
    return 0;
}

int parse_dimacs_to_edgelist(const char *base_dir, const char *file_name)
{
    using namespace std::string_literals;
    using namespace MasterThesis;
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

    unsigned num_of_vars;
    unsigned num_of_clauses;
    auto edgelist = std::make_shared<EdgelistGraph>();

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
            auto new_size = 2*num_of_vars + num_of_clauses;
            std::cout << '\t' << "Resizing Edgelist graph: " << new_size << std::endl;
            edgelist->Resize(new_size);
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
            edgelist->AddANeighbour(clause_idx, var_idx);
            // Connect the node with the clause
            edgelist->AddANeighbour(var_idx, clause_idx);
        }

        // Move on to the next clause
        ++clause_idx;
    }

    const auto output_dir = std::string{base_dir};
    const auto output_file = std::string{file_name} + ".edgelist"s;
    edgelist->SaveToFile(output_dir, output_file);

    return 0;
}