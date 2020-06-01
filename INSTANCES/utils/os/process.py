import os
import argparse
import subprocess


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


cmd_opt = argparse.ArgumentParser(description='Parser for main.py')
cmd_opt.add_argument('-wd', type=str, default='', help='Working directory. Default: Current working directory')
cmd_opt.add_argument('-printCategories', type=str2bool, nargs='?', const=True, default=True, help='Whether to print CNF files by category (True) or not (False). Default: True')
cmd_opt.add_argument('-plotFilesizes', type=str2bool, nargs='?', const=True, default=True, help='Whether to plot size of CNF files (True) or not (False). Default: True')
cmd_opt.add_argument('-plotFilteredData', type=str2bool, nargs='?', const=True, default=True, help='Whether to plot filtered CNF files (True) or not (False). Default: True')
cmd_opt.add_argument('-satzilla', type=str2bool, nargs='?', const=True, default=True, help='Whether to generate SATzilla2012 features (True) or not (False). Default: True')

cmd_args, _ = cmd_opt.parse_known_args()


def start_process(cmdpath, args, print_stdout=True):
    if not os.path.exists(cmdpath):
        raise ValueError('The command ' + os.path.abspath(cmdpath) + ' does not exist')
    cmd = [cmdpath]
    for arg in args:
        cmd.append(arg)
    process = subprocess.run(cmd, stdout=subprocess.PIPE)
    if print_stdout:
        print(process.stdout)
    return process.returncode