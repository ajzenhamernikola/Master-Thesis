import ctypes
import os


class ParsersLib(object):
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libparsers.so' % dir_path)

        self.lib.parse_dimacs_to_dgcnn_vcg.restype = ctypes.c_int
        self.lib.parse_dimacs_to_edgelist.restype = ctypes.c_int

    def parse_dimacs_to_dgcnn_vcg(self, base_dir: str, file_name: str, labels: str):
        arg0 = base_dir.encode('utf-8')
        arg1 = file_name.encode('utf-8')
        arg2 = labels.encode('utf-8')

        self.lib.parse_dimacs_to_dgcnn_vcg.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.parse_dimacs_to_dgcnn_vcg(arg0, arg1, arg2)

    def parse_dimacs_to_edgelist(self, base_dir, file_name):
        arg0 = base_dir.encode('utf-8')
        arg1 = file_name.encode('utf-8')

        self.lib.parse_dimacs_to_edgelist.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.parse_dimacs_to_edgelist(arg0, arg1)


dirname = os.path.dirname(os.path.realpath(__file__))
dll_path = f'{dirname}/build/dll/libparsers.so'
if not os.path.exists(dll_path):
    msg = f"Could not find the required shared library: libparsers.so in: {dll_path}\n" + \
          f"Try running `make` in directory: {dirname}"
    raise FileNotFoundError(msg)

parserslib = ParsersLib()
