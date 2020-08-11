#include "dcgnn_node.hpp"

namespace MasterThesis
{

std::string DcgnnNode::Tag() const
{
    return _tag;
}

void DcgnnNode::Tag(std::string val)
{
    _tag = std::move(val);
}

}