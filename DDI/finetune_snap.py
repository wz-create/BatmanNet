import argparse
from loader import MoleculeDataset, load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,precision_recall_curve,f1_score
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from splitters import scaffold_split, random_scaffold_split, random_split, scaffold_split_fp, cv_random_split
import pandas as pd

import os
import shutil
from util import *
from tensorboardX import SummaryWriter
import warnings, random

from BatmanNet.util.utils import create_logger, load_checkpoint, makedirs, build_optimizer, \
    get_task_names, save_checkpoint, build_lr_scheduler, get_loss_func
from BatmanNet.util.nn_utils import initialize_weights, param_count
from BatmanNet.data.molgraph import MolPairCollator
from BatmanNet.util.scheduler import NoamLR



warnings.filterwarnings("ignore")


# Workaround because python functions are not picklable
class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        # loss = -self.alpha*(1-input)**self.gamma*(target*torch.log(input+1e-9))-\
        #        (1-self.alpha)*input**self.gamma*((1-target)*torch.log(1-input+1e-9))
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduce=False)
        pt = torch.exp(bce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return loss

def cal_metrics(y_prob,y_true):
    auc = roc_auc_score(y_true, y_prob)
    prc = metrics.auc(metrics.precision_recall_curve(y_true,y_prob)[1],
                    metrics.precision_recall_curve(y_true,y_prob)[0])
    pred= np.int64(y_prob > 0.5)
    f1 = f1_score(y_true, pred, average='macro')
    return [auc,prc,f1]


def train(args, model, device, loader, optimizer, scheduler, loss_func, n_iter, output_wise):
    model.train()
    loss_sum, iter_count = 0, 0
    cum_loss_sum, cum_iter_count = 0, 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # batch = batch.to(device)
        _, _, batch1, batch2, features1_batch, features2_batch, mask, targets = batch
        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()
        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch1, batch2, features1_batch, features2_batch)

        loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += args.batch_size

        cum_loss_sum += loss.item()
        cum_iter_count += 1

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += args.batch_size

    return n_iter, cum_loss_sum / cum_iter_count



def eval(args, model, device, loader, loss_func):
    model.eval()
    preds = []

    # num_iters, iter_step = len(data), batch_size
    loss_sum, iter_count = 0, 0
    y_true = []
    pred_score = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        # batch = batch.to(device)
        _, _, batch1, batch2, features1_batch, features2_batch, mask, targets = batch
        class_weights = torch.ones(targets.shape)
        if next(model.parameters()).is_cuda:
            targets = targets.cuda()
            mask = mask.cuda()
            class_weights = class_weights.cuda()
        # Run model
        with torch.no_grad():
            batch_preds = model(batch1, batch2, features1_batch, features2_batch)
            iter_count += 1
            if loss_func is not None:
                loss = loss_func(batch_preds, targets) * class_weights * mask
                loss = loss.sum() / mask.sum()
                loss_sum += loss.item()
        # y=batch.y.view(pred.shape).to(torch.long).cpu().numpy()
        # pred=torch.sigmoid(pred).cpu().numpy()
        batch_preds= batch_preds.data.cpu().numpy().tolist()
        targets = targets.data.cpu().numpy().tolist()
        preds.append(batch_preds)
        y_true.append(targets)
    y_true = np.concatenate(y_true, axis=0)
    preds = np.concatenate(preds, axis=0)
    result = cal_metrics(preds, y_true)

    loss_avg = loss_sum / iter_count
    return result, loss_avg




def main():
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--parser_name', type=str, default="ddi")
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--data_path', type=str, default="../data/biosnap/raw/all.csv",
                        help='Path to data CSV file.()')
    # parser.add_argument('--features_path', type=str, nargs='*',
    #                     help='Path to features to use in FNN (instead of features_generator).')
    parser.add_argument('--features_path', type=str, default=None,
                        help='Path to features to use in FNN (instead of features_generator).')
    parser.add_argument('--features_scaling', action='store_true', default=True,
                        help='Turn off scaling of features')
    parser.add_argument('--save_dir', type=str, default="./model/biosnap",
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--dataset', type=str, default='biosnap',
                        help='root directory of dataset. For now, only classification.')

    parser.add_argument('--checkpoint_path', type=str,
                        default="../pretrained_model/model/model.ep15",
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--use_compound_names', action='store_true', default=False,
                        help='Use when test data file contains compound names in addition to SMILES strings')
    parser.add_argument('--max_data_size', type=int,
                        help='Maximum number of data points to load')
    parser.add_argument('--split_type', type=str, default='random',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'],
                        help='Method of splitting the data into train/val/test')
    parser.add_argument('--dataset_type', type=str,
                        choices=['classification', 'regression'], default='classification',
                        help='Type of dataset, e.g. classification or regression.'
                             'This determines the loss function used during training.')
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.7, 0.1, 0.2],
                        help='Split proportions for train/validation/test sets')
    parser.add_argument('--ensemble_size', type=int, default=1,
                        help='Number of models for ensemble prediction.')
    parser.add_argument('--use_pretrained_model', type=bool, default=True,
                        help='Path to model checkpoint (.pt file)')

    parser.add_argument('--self_attention', action='store_true', default=False, help='Use self attention layer. '
                                                                                     'Otherwise use mean aggregation '
                                                                                     'layer.')
    parser.add_argument('--attn_hidden', type=int, default=4, nargs='?', help='Self attention layer '
                                                                              'hidden layer size.')
    parser.add_argument('--attn_out', type=int, default=128, nargs='?', help='Self attention layer '
                                                                             'output feature size.')
    parser.add_argument('--folds_file', type=str, default=None, help='Optional file of fold labels')
    parser.add_argument('--val_fold_index', type=int, default=None, help='Which fold to use as val for leave-one-out cross val')
    parser.add_argument('--test_fold_index', type=int, default=None, help='Which fold to use as test for leave-one-out cross val')
    parser.add_argument('--features_only', action='store_true', default=False,
                        help='Use only the additional features in an FFN, no graph network')
    parser.add_argument("--output_wise", default="both", choices=["atom", "bond", "both"],
                        help="This the model parameters for pretrain model. The current finetuning task only use the "
                             "embeddings from atom branch. ")

    parser.add_argument('--ffn_hidden_size', type=int, default=200,
                        help='Hidden dim for higher-capacity FFN (defaults to hidden_size)')
    parser.add_argument('--ffn_num_layers', type=int, default=2,
                        help='Number of layers in FFN after MPN encoding')
    parser.add_argument('--init_lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--max_lr', type=float, default=1e-3,
                        help='Maximum learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4,
                        help='Final learning rate')
    parser.add_argument('--dist_coff', type=float, default=0.1, help='The dist coefficient for output of two branches.')



    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--warmup_epochs', type=float, default=2.0,
                        help='Number of epochs during which learning rate increases linearly from'
                             'init_lr to max_lr. Afterwards, learning rate decreases exponentially'
                             'from max_lr to final_lr.')

    # # parser.add_argument('--lr', type=float, default=0.0001,
    # #                     help='learning rate (default: 0.001)')
    # # parser.add_argument('--lr_decay', type=float, default=0.995,
    # #                     help='learning rate decay (default: 0.995)')
    # # parser.add_argument('--lr_scale', type=float, default=1,
    # #                     help='relative learning rate for the feature extraction layer (default: 1)')
    # # parser.add_argument('--decay', type=float, default=0,
    #                     help='weight decay (default: 0)')
    parser.add_argument('--loss_type', type=str, default="bce")


    parser.add_argument('--filename', type=str, default='', help='output filename')
    parser.add_argument('--seed', type=int, default=44, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")

    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--iters', type=int, default=1, help='number of run seeds')
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--log_freq', type=int, default=0)
    parser.add_argument('--KFold', type=int, default=5, help='number of folds for cross validation')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument('--cpu', default=False, action="store_true")
    args = parser.parse_args()

    logger = create_logger(name='train', save_dir=args.save_dir, quiet=False)
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    init_seed = args.seed
    save_dir = args.save_dir
    task_names = get_task_names(args.data_path)

    device = torch.device("cuda") if torch.cuda.is_available() and not args.cpu else torch.device("cpu")
    args.seed = args.seed
    args.runseed = args.runseed
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    exp_path = 'runs/{}/'.format(args.dataset)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    num_tasks = 1

    features_scaler, shared_dict, test_data, train_data, val_data = load_data(args, debug, logger, num_tasks)

    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)

        # Load/build model
        if args.use_pretrained_model:
            # if len(args.checkpoint_paths) == 1:
            #     cur_model = 0
            # else:
            cur_model = model_idx
            debug(f'Loading model {cur_model} from {args.checkpoint_path}')
            model = load_checkpoint(args.checkpoint_path, current_args=args, logger=logger, load_model=True)
        else:
            cur_model = model_idx
            debug(f'Building model {model_idx}')
            debug(f'Loading args of model {cur_model} from {args.checkpoint_path}')
            model = load_checkpoint(args.checkpoint_path, current_args=args, logger=logger,
                                    load_model=False)

    optimizer = build_optimizer(model, args)

    # optimizer = optim.Adam(optimizer_grouped_parameters, lr=args.lr, weight_decay=args.decay)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    # Ensure that model is saved in correct location for evaluation if 0 epochs
    # save_checkpoint(os.path.join(save_dir, 'model.pt'), model, features_scaler, args)

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, args)

    # debug(model)
    debug(f'Number of parameters = {param_count(model):,}')
    if args.cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    # Bulid data_loader
    shuffle = True
    mol_collator = MolPairCollator(shared_dict={}, args=args)
    train_loader = DataLoader(train_data,
                            batch_size=args.batch_size,
                            shuffle=shuffle,
                            num_workers=10,
                            collate_fn=mol_collator)
    val_loader = DataLoader(val_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=10,
                              collate_fn=mol_collator)
    test_loader = DataLoader(test_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=10,
                              collate_fn=mol_collator)

    loss_func = get_loss_func(args, model)

    if args.loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss_type == 'softmax':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = FocalLoss(gamma=2, alpha=0.25)

    best_result = []
    acc=0
    best_epoch, n_iter = 0, 0
    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
        train(args, model, device, train_loader, optimizer, scheduler, loss_func, n_iter, args.output_wise)
        # scheduler.step()
        print("====Evaluation")
        # if args.eval_train:
        #     train_acc = eval(args, model, device, train_loader, criterion)
        # else:
        #     print("omit the training accuracy computation")
        #     train_acc = 0
        result, val_loss = eval(args, model, device, val_loader, criterion)
        print('validation: ', result, 'val_loss:', val_loss)
        test_result, test_loss = eval(args, model, device, test_loader, criterion)
        print('test: ', test_result, 'test_loss', test_loss)
        if result[0] > acc:
            acc = result[0]
            best_result=test_result
            torch.save(model.state_dict(), exp_path + "model_seed{}.pkl".format(args.seed))
            print("save network for epoch:", epoch, acc)
    with open(exp_path+"log.txt", "a+") as f:
        log = '(No features) Test metrics: auc,prc,f1 is {}, at seed {}'.format(best_result,args.seed)
        print(log)
        f.write(log)
        f.write('\n')



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
