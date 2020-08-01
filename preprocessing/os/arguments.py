import argparse


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def validate_model_arg():
    if cmd_args.model not in ["KNN", "RF", "DGCNN"]:
        raise ValueError(f"Unknown model: {cmd_args.model}")


cmd_opt = argparse.ArgumentParser(description='Parser for main.py')
# model_output_dir, model, cnf_dir
cmd_opt.add_argument('-model_output_dir',
                     type=str,
                     default='~/Master-Thesis/out/models',
                     help='Directory to write output data to. Default: ~/Master-Thesis/out/models')
cmd_opt.add_argument('-model',
                     type=str,
                     default='DGCNN',
                     help='Model abbrevation. Must be: "KNN", "RF" or "DGCNN". Default: DGCNN')
cmd_opt.add_argument('-cnf_dir',
                     type=str,
                     default='~/Master-Thesis/INSTANCES',
                     help='Directory that contains CNF in DIMACS format. Default: ~/Master-Thesis/INSTANCES')
# Old
cmd_opt.add_argument('-wd', type=str, default='', help='Working directory. Default: Current working directory')
cmd_opt.add_argument('-printCategories', type=str2bool, nargs='?', const=True, default=True, help='Whether to print CNF files by category (True) or not (False). Default: True')
cmd_opt.add_argument('-plotFilesizes', type=str2bool, nargs='?', const=True, default=True, help='Whether to plot size of CNF files (True) or not (False). Default: True')
cmd_opt.add_argument('-plotFilteredData', type=str2bool, nargs='?', const=True, default=True, help='Whether to plot filtered CNF files (True) or not (False). Default: True')
cmd_opt.add_argument('-satzilla', type=str2bool, nargs='?', const=True, default=True, help='Whether to generate SATzilla2012 features (True) or not (False). Default: True')
cmd_opt.add_argument('-edgelist', type=str2bool, nargs='?', const=True, default=True, help='Whether to generate Edgelist files from CNF files (True) or not (False). Default: True')
cmd_opt.add_argument('-node2vec', type=str2bool, nargs='?', const=True, default=True, help='Whether to generate node2vec features from Edgelist files (True) or not (False). Default: True')
cmd_opt.add_argument('-dgcnn', type=str2bool, nargs='?', const=True, default=True, help='Whether to generate DGCNN files from CNF files (True) or not (False). Default: True')

cmd_args, _ = cmd_opt.parse_known_args()

# Arguments checks
validate_model_arg()
