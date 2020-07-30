import os 

from ..alg import process_two_arrays


def collect_roots_from_two_arrays(arr1, arr2):
    roots = []
    
    def f(inst):
        root = inst.split('/')[0]

        if root not in roots:
            roots.append(root)

    process_two_arrays(arr1, arr2, f, f)

    return roots


def collect_files_and_sizes(directory, ext, file_dict: dict):
    all_files = []
    collected_files = []
    collected_file_sizes = []
    files_not_in_dict = []
    files_not_on_disk = []

    for basedir, _, files in os.walk(directory):
        for file in files:
            if file.endswith(ext):
                filepath = os.path.join(os.path.relpath(basedir, directory), file)
                all_files.append(filepath)
                found = False 
                for key in file_dict.keys():
                    if filepath in file_dict[key]:
                        found = True
                if found:
                    collected_files.append(filepath)
                    filepath = os.path.join(basedir, file)
                    file_size = os.path.getsize(filepath)
                    collected_file_sizes.append(file_size)

    for key in file_dict.keys():
        for filepath in file_dict[key]:
            if filepath not in all_files:
                files_not_on_disk.append(filepath)

    print('Collected {} files. {} files missing from dict. {} files from dict not found on disk'.format(len(collected_files), len(files_not_in_dict), len(files_not_on_disk)))
    return collected_files, collected_file_sizes, files_not_in_dict, files_not_on_disk