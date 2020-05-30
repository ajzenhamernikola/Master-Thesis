import os 

from utils.alg import process_two_arrays


def collect_roots_from_two_arrays(arr1, arr2):
    roots = []
    
    def f(inst):
        root = inst.split('/')[0]

        if root not in roots:
            roots.append(root)

    process_two_arrays(arr1, arr2, f, f)

    return roots


def collect_files_and_sizes(directory, ext):
    collected_files = []
    collected_file_sizes = []

    for basedir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                filepath = os.path.join(os.path.relpath(basedir, directory), file)
                collected_files.append(filepath)
                filepath = os.path.join(basedir, file)
                file_size = os.path.getsize(filepath)
                collected_file_sizes.append(file_size)

    return collected_files, collected_file_sizes