import ctypes
import os

class _parsers_lib():
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.lib = ctypes.CDLL('%s/build/dll/libparsers.so' % dir_path)

        self.lib.parse_dimacs_to_dcgnn_vcg.restype = ctypes.c_int


    def parse_dimacs_to_dcgnn_vcg(self, base_dir = b'./data', file_name = b'test.cnf', label = 15):
        self.lib.parse_dimacs_to_dcgnn_vcg(base_dir, file_name, ctypes.c_uint(label))


dll_path = '%s/build/dll/libparsers.so' % os.path.dirname(os.path.realpath(__file__))
if os.path.exists(dll_path):
    PARSERSLIB = _parsers_lib()
else:
    PARSERSLIB = None
