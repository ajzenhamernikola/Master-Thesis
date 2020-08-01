#ifndef DCGNN_NODE_HPP
#define DCGNN_NODE_HPP

#include <string>

#include "node.hpp"

namespace MasterThesis
{

class DcgnnNode : public Node
{
private:
    std::string _tag;

public:
    std::string Tag() const;
    void Tag(std::string val);
};

} // namespace MasterThesis

#endif // DCGNN_NODE_HPP
