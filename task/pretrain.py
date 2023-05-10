"""
The BatmanNet pretrain function
"""
import sys
sys.path.append('../')
import time
import argparse
from argparse import Namespace
from logging import Logger
import rdkit
from tqdm import tqdm
import os
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import matplotlib.pyplot as plt
import math, random, sys
import numpy as np

from collections import deque
from torch.utils.data import DataLoader



from BatmanNet.model.models import BatmanNet_model
from BatmanNet.data.myGMAEdataset import get_data_processed, split_data, myGMAE_Collator
from BatmanNet.util.multi_gpu_wrapper import MultiGpuWrapper as mgw
from BatmanNet.util.utils import build_optimizer, build_lr_scheduler
from task.modeltrainer import ModelTrainer


def pretrain_model(args: Namespace, logger: Logger = None):
    """
        The entrey of pretrain.
        :param args: the argument.
        :param logger: the logger.
        :return:
        """
    # avoid auto optimized import by pycharm.
    s_time = time.time()
    run_training(args=args, logger=logger)
    e_time = time.time()
    print("Total Time: %.3f" % (e_time - s_time))


def run_training(args, logger):
    """
        Run the pretrain task.
        :param args:
        :param logger:
        :return:
        """

    # initalize the logger.
    if logger is not None:
        debug, _ = logger.debug, logger.info
    else:
        debug = print

    # initialize the horovod library
    if args.enable_multi_gpu:
        mgw.init()
    with_cuda = args.cuda

    train_dataset, test_dataset = get_data_processed(data_path=args.data_path, args=args)
    # train_data, test_data, _ = split_data(data=data, args=args, sizes=(0.9, 0.1, 0.0))
    args.train_data_size = len(train_dataset)
    # args.fine_tune_coff = 1.0

    shared_dict = {}
    shuffle = True
    # fg_size = 85
    # mol_collator = myGMAE_Collator(shared_dict=shared_dict, args=args)
    # Build dataloader
    train_data_dl = DataLoader(train_dataset,
                               batch_size=1,
                               shuffle=shuffle,
                               num_workers=args.num_workers,
                               collate_fn=lambda x: x[0],
                               pin_memory=True,
                               drop_last=False)
    test_data_dl = DataLoader(test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.num_workers,
                              collate_fn=lambda x: x[0],
                              pin_memory=True,
                              drop_last=False)


    # Build the embedding model.
    my_model = BatmanNet_model(args)

    #  Build the trainer.
    trainer = ModelTrainer(args=args,
                           embedding_model=my_model,
                           # atom_vocab_size=0,
                           # bond_vocab_size=0,
                           # fg_szie=fg_size,
                           train_dataloader=train_data_dl,
                           test_dataloader=test_data_dl,
                           optimizer_builder=build_optimizer,
                           scheduler_builder=build_lr_scheduler,
                           logger=logger,
                           with_cuda=with_cuda,
                           enable_multi_gpu=args.enable_multi_gpu)

    # Restore the interrupted training

    model_dir = os.path.join(args.save_dir, "model:epochs={}_batch_size={}_mask_ratio={}_V2/".format(
        args.epochs, args.batch_size, args.mask_ratio))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    plot_dir = os.path.join(args.save_dir, "plots2:epochs={}_batch_size={}_mask_ratio={}_V2".format(
        args.epochs, args.batch_size, args.mask_ratio))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    train_loss_list = []
    train_node_loss_list = []
    train_edge_loss_list = []
    val_loss_list = []
    val_node_loss_list = []
    val_edge_loss_list = []

    # Perform training
    print(f"Start training for {args.epochs} epochs")
    for epoch in range(args.epochs):

        # perform training and validation
        s_time = time.time()
        _, train_loss, train_node_loss, train_edge_loss = trainer.train(epoch)
        t_time = time.time() - s_time

        s_time = time.time()
        _, val_loss, val_node_loss, val_edge_loss = trainer.test(epoch)
        v_time = time.time() - s_time

        train_loss_list.append(train_loss)
        train_node_loss_list.append(train_node_loss)
        train_edge_loss_list.append(train_edge_loss)
        val_loss_list.append(val_loss)
        val_node_loss_list.append(val_node_loss)
        val_edge_loss_list.append(val_edge_loss)
        # print information.

        print('Epoch: {:04d}'.format(epoch),
              'loss_train: {:.6f}'.format(train_loss),
              'loss_train_node: {:.6f}'.format(train_node_loss),
              'loss_train_edge: {:.6f}'.format(train_edge_loss),
              'loss_val: {:.6f}'.format(val_loss),
              'loss_val_node: {:.6f}'.format(val_node_loss),
              'loss_val_edge: {:.6f}'.format(val_edge_loss),
              'cur_lr: {:.5f}'.format(trainer.scheduler.get_lr()[0]),
              't_time: {:.4f}s'.format(t_time),
              'v_time: {:.4f}s'.format(v_time), flush=True)

        if epoch % args.save_per_epochs == 0 or epoch + 1 == args.epochs:
            trainer.save(epoch, model_dir)

    # create model info
    model_info = "Batch Size: {}, Dropout: {}, Num Encoder Block: {}, Num Decoder Block: {}, Num Attn_head: {}".\
        format(args.batch_size, args.dropout, args.num_enc_mt_block, args.num_dec_mt_block, args.num_attn_head)



        # trainer.save_tmp(epoch, model_dir, rank)
    'figure for plotting loss'
    fig = plt.figure(0, figsize=(15, 10))
    plt.plot(np.arange(args.epochs), train_loss_list, label='Train Loss')
    plt.plot(np.arange(args.epochs), train_node_loss_list, label='Train Node Loss')
    plt.plot(np.arange(args.epochs), train_edge_loss_list, label='Train Edge Loss')
    plt.title("Train Loss" + model_info, fontsize=10, pad=15.0)
    plt.xlabel('Epoch', fontsize=20, labelpad=10)
    plt.ylabel('Loss', fontsize=20, labelpad=10)
    plt.tick_params(size=20)
    plt.legend()
    plot_name = "pretrain_train_loss"
    fig_path = os.path.join(plot_dir, plot_name + '.png')
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    fig = plt.figure(0, figsize=(15, 10))
    plt.plot(np.arange(args.epochs), val_loss_list, label='Val Loss')
    plt.plot(np.arange(args.epochs), val_node_loss_list, label='Val Node Loss')
    plt.plot(np.arange(args.epochs), val_edge_loss_list, label='Val Edge Loss')
    plt.title("Val Loss" + model_info, fontsize=10, pad=15.0)
    plt.xlabel('Epoch', fontsize=20, labelpad=10)
    plt.ylabel('Loss', fontsize=20, labelpad=10)
    plt.tick_params(size=20)
    plt.legend()
    plot_name = "pretrain_val_loss"
    fig_path = os.path.join(plot_dir, plot_name + '.png')
    plt.savefig(fig_path, dpi=200)
    plt.close(fig)

    print('over')

        # Only save final version.
    # if master_worker:
    #     trainer.save(args.epochs, model_dir, "")


if __name__ == '__main__':
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train', required=True)
    # parser.add_argument('--vocab', required=True)
    # parser.add_argument('--save_dir', required=True)
    parser.add_argument('--data_path', type=str, default="../data/processed")
    parser.add_argument('--save_dir', type=str, default="../pretrained_model")
    parser.add_argument('--hidden_size', type=int, default=100)
    # parser.add_argument('--latent_size', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--depth', type=int, default=3)
    #parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_enc_mt_block', type=int, default=6)
    parser.add_argument('--num_dec_mt_block', type=int, default=2)
    parser.add_argument('--num_attn_head', type=int, default=2)

    parser.add_argument('--dropout', type=int, default=0.1)
    parser.add_argument('--activation', type=str, default='PReLU')
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--undirected', type=bool, default=False)
    parser.add_argument('--dense', type=bool, default=True)
    # parser.add_argument('--embedding_output_type', type=str, default='both')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_per_epochs', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--enable_multi_gpu', type=bool, default=False)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--dist_coff', type=int, default=0)
    parser.add_argument('--init_lr', type=float, default=0.0002)
    parser.add_argument('--max_lr', type=float, default=0.0004)
    parser.add_argument('--final_lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.0000001)
    parser.add_argument('--mask_ratio', type=float, default=0.7)

    args = parser.parse_args()
    print(args)
    print(torch.cuda.is_available())

    pretrain_model(args)



