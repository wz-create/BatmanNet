import sys
sys.path.append('../')
import torch
import argparse
from multiprocessing import Pool
import numpy as np
import os
from tqdm import tqdm
import time
import math, random
import pickle
from BatmanNet.data.molgraph import mol2graph
from BatmanNet.data.datautils import split_data, node_random_masking, edge_random_masking, get_mask_target_features

import rdkit

def data_process(smiles_batch, mask_ratio):
    shared_dict = {}
    graph_batch = mol2graph(smiles_batch, shared_dict).get_components()
    node_masked_batch = node_random_masking(graph_batch, mask_ratio)
    edge_masked_batch = edge_random_masking(graph_batch, mask_ratio)
    mask_node_target, mask_edge_target = get_mask_target_features(node_masked_batch, edge_masked_batch)

    return (graph_batch, node_masked_batch, edge_masked_batch, mask_node_target, mask_edge_target)


def convert(data_path, pool, output_path, batch_size, mask_ratio, njobs, shuffle):
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)
    
    out_path = os.path.join(output_path, 'batch_size={}_mask_ratio={}'.format(batch_size, mask_ratio))
    if os.path.isdir(out_path) is False:
        os.makedirs(out_path)

    print('Input File read')
    with open(data_path) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    if shuffle:
        random.shuffle(data)
    smiles_batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]
    if len(smiles_batches[-1]) < batch_size:
        smiles_batches.pop()

    train_smiles_batches, test_smiles_batches, _ = split_data(data=smiles_batches, sizes=(0.9, 0.1, 0.0))

    print('Data processing......')
    s_time = time.time()

    # 使用for循环处理数据
    train_processed_list = []
    for smiles_batch in tqdm(train_smiles_batches):
        train_processed = data_process(smiles_batch, mask_ratio)
        train_processed_list.append(train_processed)

    test_processed_list = []
    for smiles_batch in tqdm(test_smiles_batches):
        test_processed = data_process(smiles_batch, mask_ratio)
        test_processed_list.append(test_processed)

    # 使用多线程处理数据
    # data_processed_list = pool.map(data_process, smiles_batches, mask_ratio)
    # pool.close()
    # pool.join()

    print("train_len:{}".format(len(train_processed_list)))
    print("test_len:{}".format(len(test_processed_list)))
    print('Data processing Complete')
    d_time = time.time() - s_time
    print('Data processing time:{}min'.format(d_time / 60))

    with open(os.path.join(out_path, 'train_mol_graph.pkl'), 'wb') as f:
        pickle.dump(train_processed_list, f)
    with open(os.path.join(out_path, 'test_mol_graph.pkl'), 'wb') as f:
        pickle.dump(test_processed_list, f)

    return True


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default="../data/zinc/all.txt")
    parser.add_argument('--output_path', type=str, default="../data/processed")
    parser.add_argument('--njobs', type=int, default=40, help="8")
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mask_ratio', type=float, default=0.6)

    args = parser.parse_args()

    args.njobs = int(args.njobs)

    pool = Pool(args.njobs)

    convert(args.data_path, pool, args.output_path, args.batch_size, args.mask_ratio, args.njobs, shuffle=args.shuffle)
