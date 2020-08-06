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
    if cmd_args.model not in ["KNN", "RF", "GCN", "DGCNN"]:
        raise ValueError(f"Unknown model: {cmd_args.model}")


cmd_opt = argparse.ArgumentParser(description='Parser for main.py')

# New
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

# DGCNN
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='DGCNN', help='gnn model to use')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-attr_dim', type=int, default=0, help='dimension of attributes')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-look_behind', type=int, default=50, help='number of epochs to train before early stopping check')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-edge_feat_dim', type=int, default=0, help='dimension of edge features')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0,
                     help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='64', help='dimension(s) of latent layers')
cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-conv1d_activation', type=str, default='ReLU', help='which nn activation layer to use')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='graph embedding output size')
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of mlp hidden layer')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
cmd_opt.add_argument('-print_auc', type=bool, default=False,
                     help='whether to print AUC (for binary classification only)')
cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')

cmd_args, _ = cmd_opt.parse_known_args()

cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]

# Arguments checks
validate_model_arg()
