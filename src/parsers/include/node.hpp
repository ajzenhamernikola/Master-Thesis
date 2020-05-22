#ifndef NODE_HPP
#define NODE_HPP

#include <string>
#include <set>

namespace master_thesis
{
    using _data_type = unsigned;
    using _node_tag = std::string;

    template<typename T>
    struct node 
    {
        _node_tag tag;
        std::set<T> neighbours;
    };
    
} // namespace master_thesis

#endif // NODE_HPP
