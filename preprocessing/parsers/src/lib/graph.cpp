#include "graph.hpp"

namespace MasterThesis
{

void Graph::Resize(const unsigned num_of_nodes)
{
    for (auto i = 0u; i < num_of_nodes; ++i)
    {
        _nodes.push_back(std::make_shared<Node>());
    }
}

void Graph::AddANeighbour(const unsigned node, unsigned neighbour)
{
    _nodes[node]->Neighbours().insert(std::move(neighbour));
}

}