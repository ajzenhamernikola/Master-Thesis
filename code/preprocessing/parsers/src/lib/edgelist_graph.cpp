#include <algorithm>
#include <iterator>
#include <utility>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "edgelist_graph.hpp"

namespace MasterThesis
{

void EdgelistGraph::Resize(const unsigned num_of_nodes)
{
    Graph::Resize(num_of_nodes);
    for (auto i = 0u; i < _nodes.size(); ++i)
    {
        _nodes[i]->Neighbours().insert(std::move(i));
    }
}

void EdgelistGraph::SaveToFile(const std::string & base_dir, const std::string & file_name)
{
    using namespace std::string_literals;
    namespace fs = std::filesystem;

    if (!fs::exists(base_dir))
    {
        fs::create_directories(base_dir);
    }

    auto output_file = base_dir + "/"s + file_name;

    std::cout << "Saving parsed data to file: " << output_file << std::endl;
    std::ofstream output{ output_file };

    for (auto i = 0u; i < _nodes.size(); ++i)
    {
        auto g_node = std::move(_nodes[i]);
        auto & neighbours = g_node->Neighbours();
        std::for_each(std::cbegin(neighbours), std::cend(neighbours), [&](const auto neighbour)
        {
            output << i << " " << neighbour << std::endl;
        });
    }
}

}