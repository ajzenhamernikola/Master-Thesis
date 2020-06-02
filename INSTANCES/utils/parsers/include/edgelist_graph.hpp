#ifndef EDGELIST_MATRIX_HPP
#define EDGELIST_MATRIX_HPP

#include <string>

#include "graph.hpp"

namespace MasterThesis
{

class EdgelistGraph : public Graph
{
public:
    void SaveToFile(const std::string & base_dir, const std::string & file_name) override;
};

}

#endif // EDGELIST_MATRIX_HPP
