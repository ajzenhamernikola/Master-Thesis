#ifndef NODE_HPP
#define NODE_HPP

#include <string>
#include <set>

namespace MasterThesis
{

class Node
{
    std::set<unsigned> _neighbours;

public:
    virtual ~Node() = default;
    
    std::set<unsigned> & Neighbours();
};

} // namespace master_thesis

#endif // NODE_HPP
