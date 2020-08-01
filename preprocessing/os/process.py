import os
import subprocess


def start_process(cmdpath, args, absolute=False, return_stdout=True):
    if not absolute and not os.path.exists(cmdpath):
        raise ValueError('The command ' + os.path.abspath(cmdpath) + ' does not exist')
    cmd = [cmdpath]
    for arg in args:
        cmd.append(arg)
    process = subprocess.run(cmd)
    if return_stdout:
        return process.returncode, process.stdout
    return process.returncode