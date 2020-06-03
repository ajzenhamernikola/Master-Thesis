#ifndef LIBPARSERS_H
#define LIBPARSERS_H

extern "C" int parse_dimacs_to_dcgnn_vcg(const char *base_dir, const char *file_name, const unsigned label);
extern "C" int parse_dimacs_to_edgelist(const char *base_dir, const char *file_name);

#endif // LIBPARSERS_H
