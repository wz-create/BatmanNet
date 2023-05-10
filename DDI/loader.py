import os,re,glob,copy
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
import sqlite3
import torch.utils.data
from torch_geometric.data import Data, Batch

import torch

from argparse import Namespace
from logging import Logger
from typing import List, Set, Tuple, Union, Dict


from BatmanNet.data.moldataset import MoleculePairDatapoint, MoleculeDataset, StandardScaler

from BatmanNet.util.utils import load_pair_features, split_data, get_class_sizes, get_task_names



def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A MoleculeDataset.
    :return: A MoleculeDataset with only valid molecules.
    """
    datapoint_list = []
    for idx, datapoint in enumerate(data):
        if datapoint.smiles1 == '':
            print(f'invalid smiles1 {idx}: {datapoint.smiles1}')
            continue
        mol1 = Chem.MolFromSmiles(datapoint.smiles1)
        if mol1 == None:
            print(f'invalid smiles1 {idx}: {datapoint.smiles1}')
            continue
        if mol1.GetNumHeavyAtoms() == 0:
            print(f'invalid heavy {idx}')
            continue

        if datapoint.smiles2 == '':
            print(f'invalid smiles2 {idx}: {datapoint.smiles2}')
            continue
        mol2 = Chem.MolFromSmiles(datapoint.smiles2)
        if mol2 == None:
            print(f'invalid smiles2 {idx}: {datapoint.smiles2}')
            continue
        if mol2.GetNumHeavyAtoms() == 0:
            print(f'invalid heavy {idx}')
            continue

        datapoint_list.append(datapoint)
    return MoleculeDataset(datapoint_list)



def load_data(args, debug, logger, num_tasks):
    """
    load the training data.
    :param args:
    :param debug:
    :param logger:
    :return:
    """
    # Get data
    debug('Loading data')
    # args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, dataset_name=args.dataset, args=args, logger=logger)

    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)
    else:
        args.features_dim = 0
    shared_dict = {}
    args.num_tasks = num_tasks
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

    if args.dataset == 'twosides' and args.KFold == 5:
        return data


    if args.dataset == 'biosnap' or args.dataset == 'twosides':
        # Split data
        debug(f'Splitting data with seed {args.seed}')

        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type,
                                                         sizes=args.split_sizes, seed=args.seed, args=args, logger=logger)

        # class_sizes = get_class_sizes(data)  # 确定分类数据集中不同类别的比例
        # debug('Class sizes')
        # for i, task_class_sizes in enumerate(class_sizes):
        #     debug(f'{args.task_names[i]} '
        #           f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

        if args.features_scaling:
            features_scaler = train_data.normalize_features(replace_nan_token=0)
            val_data.normalize_features(features_scaler)
            test_data.normalize_features(features_scaler)
        else:
            features_scaler = None

        args.train_data_size = len(train_data)
        debug(f'Total size = {len(data):,} | '
              f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

        return features_scaler, shared_dict, test_data, train_data, val_data


def get_data(path: str,
             dataset_name:str,
             skip_invalid_smiles: bool = True,
             args: Namespace = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = None,
             logger: Logger = None) -> MoleculeDataset:
    """
    Gets smiles string and target values (and optionally compound names if provided) from a CSV file.

    :param path: Path to a CSV file.
    :param skip_invalid_smiles: Whether to skip and filter out invalid smiles.
    :param args: Arguments.
    :param features_path: A list of paths to files containing features. If provided, it is used
    in place of args.features_path.
    :param max_data_size: The maximum number of data points to load.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :param logger: Logger.
    :return: A MoleculeDataset containing smiles strings and target values along
    with other info such as additional features and compound names when desired.
    """
    debug = logger.debug if logger is not None else print

    if args is not None:
        # Prefer explicit function arguments but default to args if not provided
        features_path = features_path if features_path is not None else args.features_path
        max_data_size = max_data_size if max_data_size is not None else args.max_data_size
        use_compound_names = use_compound_names if use_compound_names is not None else args.use_compound_names
    else:
        use_compound_names = False

    max_data_size = max_data_size or float('inf')
    # Load features
    if features_path is not None:
        features_data1 = []
        features_data2 = []
        # for feat_path in features_path:
        #     features_data.append(load_features(feat_path))  # each is num_data x num_features
        feature1, feature2 = load_pair_features(features_path)
        features_data1.append(feature1)
        features_data2.append(feature2)

        features_data1 = np.concatenate(features_data1, axis=1)
        features_data2 = np.concatenate(features_data2, axis=1)
        args.features_dim = len(features_data1[0])
    else:
        features_data1 = None
        features_data2 = None
        if args is not None:
            args.features_dim = 0

    if dataset_name == 'twosides':
        conn = sqlite3.connect(path)
        drug = pd.read_sql("select * from Drug", conn)
        idToSmiles = {}
        for i in range(drug.shape[0]):
            idToSmiles[drug.loc[i][0]] = drug.loc[i][3]
        smile = drug['smile']
        # positive_path = os.path.join('../data/twosides/raw/', 'twosides_interactions.csv')
        # negative_path = os.path.join(self.root, 'raw', 'reliable_negatives.csv')
        positive_path = '../data/twosides/raw/twosides_interactions.csv'
        negative_path = '../data/twosides/raw/reliable_negatives.csv'
        positive = pd.read_csv(positive_path, header=None)
        negative = pd.read_csv(negative_path, header=None)
        df = pd.concat([positive, negative])

        drugs1 = []
        drugs2 = []
        labels = []
        for i in tqdm(range(df.shape[0])):
            try:
                data1 = idToSmiles[df.iloc[i][0]]
                data2 = idToSmiles[df.iloc[i][1]]
                target = df.iloc[i][2]
                drugs1.append(data1)
                drugs2.append(data2)
                labels.append(float(target))

            except:
                continue
        drugs1 = np.array(drugs1)
        drugs2 = np.array(drugs2)
        labels = np.array(labels)

        assert len(drugs1) == len(drugs2) == len(labels)

        lines = list(list(t) for t in zip(drugs1, drugs2, labels))

        # data = MoleculeDataset([
        #     MoleculePairDatapoint(
        #         line=line,
        #         args=args,
        #         features1=features_data1[i] if features_data1 is not None else None,
        #         features2=features_data2[i] if features_data2 is not None else None,
        #         use_compound_names=use_compound_names
        #     ) for i, line in tqdm(enumerate(lines), total=len(lines), disable=True)
        # ])
        data = [
            MoleculePairDatapoint(
                line=line,
                args=args,
                features1=features_data1[i] if features_data1 is not None else None,
                features2=features_data2[i] if features_data2 is not None else None,
                use_compound_names=use_compound_names
            ) for i, line in tqdm(enumerate(lines), total=len(lines), disable=True)
        ]

        # Filter out invalid SMILES
        if skip_invalid_smiles:
            original_data_len = len(data)
            data = filter_invalid_smiles(data)

            if len(data) < original_data_len:
                debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')


    if dataset_name == 'biosnap':

        df = pd.read_csv(path)
        # positive = pd.read_csv(trn)
        # negative = pd.read_csv(test)
        # df = pd.concat([positive, negative])

        drugs1 = df['Drug1_SMILES'].values
        drugs2 = df['Drug2_SMILES'].values
        labels = df['label'].values

        assert len(drugs1) == len(drugs2) == len(labels)

        lines = list(list(t) for t in zip(drugs1, drugs2, labels))

        data = MoleculeDataset([
            MoleculePairDatapoint(
                line=line,
                args=args,
                features1=features_data1[i] if features_data1 is not None else None,
                features2=features_data2[i] if features_data2 is not None else None,
                use_compound_names=use_compound_names
            ) for i, line in tqdm(enumerate(lines), total=len(lines), disable=True)
        ])

        # Filter out invalid SMILES
        if skip_invalid_smiles:
            original_data_len = len(data)
            data = filter_invalid_smiles(data)

            if len(data) < original_data_len:
                debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')
    return data




# test MoleculeDataset object
if __name__ == "__main__":

    # create_all_datasets()
    from torch_geometric.data import DataLoader
    from dataloader import DataLoaderMasking
    from util import *
    from splitters import *
    import warnings
    warnings.filterwarnings("ignore")

    transform = Compose([Random_graph(),
                         MaskAtom(num_atom_type=119, num_edge_type=5, mask_rate=0.15,
                                  mask_edge=0.15),
                         Add_seg(),
                         Add_collection_node(num_atom_type=119, bidirection=False)])
    dataset = PretrainDataset("data/pretraining",transform=transform)
    # # loader = DataLoader(dataset,batch_size=10,shuffle=True)
    # # for i,d in enumerate(loader):
    # #
    # #     print(d)
    # #     if i>10:
    # #         break
    #
    #
    # # data = dataset.get(0)
    # # num_node = data.num_nodes
    # # print(data)
    # # print(num_node)
    # smiles_list = \
    # pd.read_csv('/data/lpy/pretrain_dataset/dataset/' + 'bbbp' + '/processed/smiles.csv', header=None)[0].tolist()
    # train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,
    #                                                             frac_valid=0.1, frac_test=0.1)
    # # print(train_dataset.indices())
    # # print(valid_dataset.indices())
    # print(valid_dataset.get(0))
    # print(valid_dataset.get(1083))
    # print(dataset.get(0))
    # print(dataset.get(1083))
    #
    # print(len(train_dataset),len(valid_dataset),len(test_dataset))

    # train_loader = DataLoaderMasking(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                                  num_workers=args.num_workers)
    # val_loader = DataLoaderMasking(valid_dataset, batch_size=args.batch_size, shuffle=False,
    #                                num_workers=args.num_workers)
    # test_loader = DataLoaderMasking(test_dataset, batch_size=args.batch_size, shuffle=False,
    #                                 num_workers=args.num_workers)


