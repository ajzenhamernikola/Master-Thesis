#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <utility>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <memory>

#include "node.hpp"

namespace MasterThesis
{

class Graph
{
protected:
    std::vector<std::shared_ptr<Node>> _nodes;

public:
    virtual ~Graph() = default;

    void AddANeighbour(const unsigned node, unsigned neighbour);

    virtual void Resize(const unsigned num_of_nodes);
    virtual void SaveToFile(const std::string & base_dir, const std::string & file_name) = 0;
};

}

#endif // GRAPH_HPP
