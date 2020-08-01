#include <vector>
#include <utility>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <algorithm>
#include <memory>
#include <sstream>

#include "dgcnngraph.hpp"

namespace MasterThesis
{

DGCNNGraph::DGCNNGraph(const char *labels)
    : Graph()
{
    std::istringstream in{std::string(labels)};
    std::string label;
    while (in >> label)
    {
        _labels.push_back(label);
    }
}

void DGCNNGraph::Resize(const unsigned num_of_nodes)
{
    for (auto i = 0u; i < num_of_nodes; ++i) 
    {
        _nodes.push_back(std::make_shared<DcgnnNode>());
    }
    for (auto i = 0u; i < _nodes.size(); ++i)
    {
        _nodes[i]->Neighbours().insert(std::move(i));
    }
}

void DGCNNGraph::SetTagForNode(const unsigned node, std::string tag)
{
    std::dynamic_pointer_cast<DcgnnNode>(_nodes[node])->Tag(std::move(tag));
}

void DGCNNGraph::SaveToFile(const std::string & base_dir, const std::string & file_name)
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
    output << _nodes.size() << " ";
    std::copy(std::cbegin(_labels), std::cend(_labels), std::ostream_iterator<std::string>(output, " "));
    output << std::endl;
    std::for_each(std::cbegin(_nodes), std::cend(_nodes), [&](const auto g_node)
    {
        auto & neighbours = g_node->Neighbours();
        auto g_node_typed = std::dynamic_pointer_cast<DcgnnNode>(g_node);
        output << g_node_typed->Tag() << " " << neighbours.size() << " ";
        std::copy(std::cbegin(neighbours), std::cend(neighbours), std::ostream_iterator<unsigned>(output, " "));
        output << std::endl;
    });
}

}