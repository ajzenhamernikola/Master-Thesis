#ifndef LIBPARSERS_H
#define LIBPARSERS_H

extern "C" int parse_dimacs_to_dgcnn_vcg(const char *base_dir, const char *file_name, const char *labels);
extern "C" int parse_dimacs_to_edgelist(const char *base_dir, const char *file_name);

#endif // LIBPARSERS_H
