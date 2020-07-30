#ifndef DCGNN_MATRIX_HPP
#define DCGNN_MATRIX_HPP

#include <string>
#include <vector>

#include "graph.hpp"
#include "dcgnn_node.hpp"

namespace MasterThesis
{

class DGCNNGraph : public Graph
{
private:
    std::vector<std::string> _labels;

public:
    DGCNNGraph(const char *labels);

    void Resize(const unsigned num_of_nodes) override;
    void SaveToFile(const std::string & base_dir, const std::string & file_name) override;
    
    void SetTagForNode(const unsigned node, std::string tag);
};

}

#endif // DCGNN_MATRIX_HPP
