import os,pickle
import numpy as np
from args import *
from model import *
from trainer import Trainer
from dataset import load_dataset_random
from utils import *
import warnings
from argparse import Namespace
from BatmanNet.util.utils import get_model_args
warnings.filterwarnings("ignore")

def load_args(path: str,
              current_args: Namespace = None,
              ):
    """
    Loads a model args.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :return: The loaded args.
    """
    # Load args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    model_ralated_args = get_model_args()

    if current_args is not None:
        for key, value in vars(args).items():
            if key in model_ralated_args:
                setattr(current_args, key, value)
    else:
        current_args = args
    return current_args

def load_checkpoint(path: str,
                    current_args: Namespace = None,
                    cuda: bool = None,
                    n_word = None,
                    load_model = True):
    """
    Loads a model checkpoint.

    :param path: Path where checkpoint is saved.
    :param current_args: The current arguments. Replaces the arguments loaded from the checkpoint if provided.
    :param cuda: Whether to move model to cuda.
    :param logger: A logger.
    :return: The loaded MPNN.
    """
    # Load model and args
    state = torch.load(path, map_location=lambda storage, loc: storage)
    args, loaded_state_dict = state['args'], state['state_dict']
    model_ralated_args = get_model_args()

    if current_args is not None:
        for key, value in vars(args).items():
            if key in model_ralated_args:
                setattr(current_args, key, value)
    else:
        current_args = args

    # args.cuda = cuda if cuda is not None else args.cuda

    # Build model
    model = CompoundProteinInteractionPrediction(args=current_args, n_word=n_word)
    if load_model == False:
        return model.cuda()
    else:
        model_state_dict = model.state_dict()

        # Skip missing parameters and parameters of mismatched size
        pretrained_state_dict = {}
        for param_name in loaded_state_dict.keys():
            new_param_name = param_name
            if new_param_name not in model_state_dict:
                print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
            elif model_state_dict[new_param_name].shape != loaded_state_dict[param_name].shape:
                print(f'Pretrained parameter "{param_name}" '
                      f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                      f'model parameter of shape {model_state_dict[new_param_name].shape}.')
            else:
                # debug(f'Loading pretrained parameter "{param_name}".')
                pretrained_state_dict[new_param_name] = loaded_state_dict[param_name]
        # Load pretrained weights
        model_state_dict.update(pretrained_state_dict)
        model.load_state_dict(model_state_dict)

        if cuda:
            print('Moving model to cuda')
            model = model.cuda()

        return model


def run_training(args):
    seed_set(args.seed)
    train_dataset, valid_dataset, test_dataset, n_word = load_dataset_random(args, args.data_path, args.dataset, args.seed)
    # args.node_in_dim = train_dataset.num_node_features
    # args.edge_in_dim = train_dataset.num_edge_features

    option = args.__dict__
    # n_word = train_dataset.n_word
    # current_args = load_args(args.checkpoint_path, current_args=args)
    print(f'Building model')
    print(f'Loading model from {args.checkpoint_path}')
    model = load_checkpoint(args.checkpoint_path, current_args=args, n_word=n_word, load_model=True)


    # model.from_pretrain(args.checkpoint_path)
    trainer = Trainer(args, model, train_dataset, valid_dataset, test_dataset)

    test_auc = trainer.train()
    return test_auc


def cross_validate(args):
    """k-fold cross validation"""
    # Initialize relevant variables
    init_seed = args.seed
    root_save_dir = args.exp_path
    # Run training on different random seeds for each fold
    all_scores = []
    for fold_num in range(args.num_folds):
        print('Fold {}'.format(fold_num))
        args.seed = init_seed + fold_num
        args.exp_path = os.path.join(root_save_dir, 'seed_{}'.format(args.seed))
        if not os.path.exists(args.exp_path):
            os.makedirs(args.exp_path)
        model_scores = run_training(args)
        all_scores.append(model_scores)
    all_scores = np.array(all_scores)

    # Report scores for each fold
    save_print_log('=='*20, root_save_dir)
    for fold_num, scores in enumerate(all_scores):
        msg = 'Seed {} ==> {} = {}'.format(init_seed+fold_num,args.metric,scores)
        save_print_log(msg,root_save_dir)

    # Report scores across models
    mean_score, std_score = np.nanmean(all_scores), np.nanstd(all_scores)  # average score for each model across tasks
    msg = 'Overall test {} = {} +/- {}'.format(args.metric,mean_score,std_score)
    save_print_log(msg, root_save_dir)
    return mean_score, std_score
if __name__ == '__main__':
    cross_validate(args)