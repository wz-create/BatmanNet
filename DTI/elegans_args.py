import argparse
import os
import random
import time

class Option(object):
    def __init__(self, d):
        self.__dict__ = d


parser = argparse.ArgumentParser()
parser.add_argument('--parser_name', type=str, default="dti")
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--model', type=str, default='human')
parser.add_argument("--data_path", type=str, default='../data/celegans/raw/data.txt', help="all data dir")
parser.add_argument("--dataset", type=str, default='celegans')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument("--gpu", type=str, nargs='+', default='1', help="CUDA device ids")
parser.add_argument("--metric", type=str, default='roc')

parser.add_argument('--features_path', type=str, default=None,
                        help='Path to features to use in FNN (instead of features_generator).')
parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
parser.add_argument('--max_data_size', type=int, help='Maximum number of data points to load')
parser.add_argument('--checkpoint_path', type=str,
                        default="../pretrained_model/model/model.ep15",
                        help='Path to model checkpoint (.pt file)')
parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression'], default='classification',
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')
parser.add_argument('--self_attention', action='store_true', default=False, help='Use self attention layer. '
                                                                                     'Otherwise use mean aggregation '
                                                                                     'layer.')
parser.add_argument('--ffn_hidden_size', type=int, default=200,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')

parser.add_argument("--hid", type=int, default=32, help="hidden size of transformer model")
parser.add_argument('--heads', default=4, type=int)
parser.add_argument('--depth', default=1, type=int)
parser.add_argument("--dropout", type=float, default=0.2)

parser.add_argument('--num_folds', default=3, type=int)
parser.add_argument('--minimize_score', default=False, action="store_true")
parser.add_argument('--num_iters',default=3,type=int)

parser.add_argument("--batch_size", type=int, default=8, help="number of batch_size")
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument("--num_workers", type=int, default=4, help="dataloader worker size")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate of adam")
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--clip_norm", type=float, default=0.0)
parser.add_argument('--lr_scheduler_patience', default=20, type=int)
parser.add_argument('--early_stop_patience', default=-1, type=int)
parser.add_argument('--lr_decay', default=0.98, type=float)
parser.add_argument('--focalloss', default=False, action="store_true")
parser.add_argument('--single_task_mseloss', default=False)

parser.add_argument('--model_dir', default=None, type=str)
parser.add_argument('--eval', default=False, action="store_true")
parser.add_argument("--exps_dir", default='test',type=str, help="out/")
parser.add_argument('--exp_name', default=None, type=str)
parser.add_argument('--test_path', default=None, type=str)
parser.add_argument('--cpu', default=False, action="store_true")
parser.add_argument("--fold", type=int, default=0)
d = vars(parser.parse_args())
args = Option(d)

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))


if len(args.gpu) > 1:
    args.parallel = True
    args.parallel_devices = args.gpu
else:
    args.parallel = False
    args.parallel_devices = args.gpu

if args.exp_name is None:
    args.tag = time.strftime("%m-%d-%H-%M")
else:
    args.tag = args.exp_name
args.tag = args.dataset + args.tag
args.exp_path = os.path.join(args.exps_dir, args.tag)
if not os.path.exists(args.exp_path):
    os.makedirs(args.exp_path)
args.code_file_path = os.path.abspath(__file__)
