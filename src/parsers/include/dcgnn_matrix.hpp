#ifndef DCGNN_MATRIX_HPP
#define DCGNN_MATRIX_HPP

#include <vector>
#include <utility>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>

#include "node.hpp"

namespace master_thesis
{

template<typename T>
class dcgnn_matrix
{
private:
    std::vector<node<T>> _data;
    unsigned _label;
public:
    dcgnn_matrix(const unsigned label);

    void resize(const _data_type num_of_nodes);
    void add_a_neighbour(const _data_type node, T neighbour);
    void set_tag_for_node(const _data_type node, _node_tag tag);
    void save_to_file(const std::string & base_dir, const std::string & file_name);
};

template<typename T>
dcgnn_matrix<T>::dcgnn_matrix(const unsigned label)
    : _label(label)
{}

template<typename T>
void dcgnn_matrix<T>::resize(const _data_type num_of_nodes)
{
    _data.resize(num_of_nodes);
}

template<typename T>
void dcgnn_matrix<T>::add_a_neighbour(const _data_type node, T neighbour)
{
    _data[node].neighbours.insert(std::move(neighbour));
}

template<typename T>
void dcgnn_matrix<T>::set_tag_for_node(const _data_type node, _node_tag tag)
{
    _data[node].tag = std::move(tag);
}

template<typename T>
void dcgnn_matrix<T>::dcgnn_matrix::save_to_file(const std::string & base_dir, const std::string & file_name)
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
    output << _data.size() << " " << _label << std::endl;
    std::for_each(std::cbegin(_data), std::cend(_data), [&](const auto & g_node)
    {
        output << g_node.tag << " " << g_node.neighbours.size() << " ";
        std::copy(std::cbegin(g_node.neighbours), std::end(g_node.neighbours), std::ostream_iterator<_data_type>(output, " "));
        output << std::endl;
    });
}

}

#endif // DCGNN_MATRIX_HPP
