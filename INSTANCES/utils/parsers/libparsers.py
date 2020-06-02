import ctypes
import os

class _parsers_lib():
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libparsers.so' % dir_path)

        self.lib.parse_dimacs_to_dcgnn_vcg.restype = ctypes.c_int
        self.lib.parse_dimacs_to_edgelist.restype = ctypes.c_int


    def parse_dimacs_to_dcgnn_vcg(self, base_dir, file_name, label):
        arg0 = base_dir.encode('utf-8')
        arg1 = file_name.encode('utf-8')
        arg2 = ctypes.c_uint(label)

        self.lib.parse_dimacs_to_dcgnn_vcg.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        self.lib.parse_dimacs_to_dcgnn_vcg(arg0, arg1, arg2)


    def parse_dimacs_to_edgelist(self, base_dir, file_name):
        arg0 = base_dir.encode('utf-8')
        arg1 = file_name.encode('utf-8')

        self.lib.parse_dimacs_to_edgelist.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.lib.parse_dimacs_to_edgelist(arg0, arg1)


dll_path = '%s/build/dll/libparsers.so' % os.path.dirname(os.path.realpath(__file__))
if os.path.exists(dll_path):
    PARSERSLIB = _parsers_lib()
else:
    PARSERSLIB = None
