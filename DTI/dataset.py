import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset
import torch_geometric.transforms as T
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, MolFromSmiles, AllChem
from collections import defaultdict
import pickle
import sys
from utils import *
from mol2graph import *
import re
from typing import List
from BatmanNet.util.utils import load_features
from BatmanNet.data.moldataset import MoleculeProteinDatapoint, MoleculeDataset


class Get_DTI_Data:
    def __init__(self, args, root, dataset_name):

        self.dataset = dataset_name
        with open('../data/3ngram_vocab', 'r') as f:
            word_dic = f.read().split('\n')
            if word_dic[-1] == '':
                word_dic.pop()
        # word_dic=['pad']+['other{}'.format(n) for n in range(10)]+word_dic
        self.word_dict = {}
        for i, item in enumerate(word_dic):
            self.word_dict[item] = i
        self.n_word = len(self.word_dict)

        data = self.get_dti_data(args=args, root=root, dataset_name=dataset_name)
        self.data = data

    def get_dti_data(self, args, root, dataset_name,
                     skip_invalid_smiles: bool = True,
                     features_path: List[str] = None,
                     max_data_size: int = None,
                     use_compound_names: bool = None
                     ):
        print('Loading data')
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
            features_data = []
            # for feat_path in features_path:
            #     features_data.append(load_features(feat_path))  # each is num_data x num_features
            feature = load_features(features_path)
            features_data.append(feature)

            features_data = np.concatenate(features_data, axis=1)
            args.features_dim = len(features_data[0])
        else:
            features_data = None
            if args is not None:
                args.features_dim = 0

        if dataset_name == 'human':
            with open(root, 'r') as f:
                data_list = f.read().strip().split('\n')
            data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
            N = len(data_list)

            ngram = 3
            positive = 0
            molecules = []
            proteins = []
            protein_lens = []
            labels = []
            for no, data in enumerate(tqdm(data_list)):
                smiles, sequence, interaction = data.strip().split()

                # count positive samples
                positive += int(interaction)
                words = self.split_sequence(sequence, ngram)

                molecules.append(smiles)
                labels.append(torch.LongTensor([int(interaction)]))
                proteins.append(torch.LongTensor(words))
                protein_lens.append(torch.LongTensor([len(words)]))

                # mol = MolFromSmiles(smiles)
                # if mol == None:
                #     continue
                # data = mol_to_graph_data_dic(mol)
                # data.y = torch.LongTensor([int(interaction)])
                # words = split_sequence(sequence, ngram)
                # data.protein = torch.LongTensor(words)
                # data.pr_len = torch.LongTensor([len(words)])
                # DATALIST.append(data)

            assert len(molecules) == len(proteins) == len(labels)

            lines = list(list(t) for t in zip(molecules, proteins, protein_lens, labels))

            data = [
                MoleculeProteinDatapoint(
                    line=line,
                    args=args,
                    features=features_data if features_data is not None else None,
                    use_compound_names=use_compound_names
                ) for i, line in tqdm(enumerate(lines), total=len(lines), disable=True)
            ]

            weights = [len(data) / (len(data) - positive), len(data) / positive]
            print(weights)

            # Filter out invalid SMILES
            if skip_invalid_smiles:
                original_data_len = len(data)
                data = self.filter_invalid_smiles(data)

                if len(data) < original_data_len:
                    print(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

        if dataset_name == 'celegans':
            with open(root, 'r') as f:
                data_list = f.read().strip().split('\n')
            data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
            N = len(data_list)

            ngram = 3
            positive = 0
            molecules = []
            proteins = []
            protein_lens = []
            labels = []
            for no, data in enumerate(tqdm(data_list)):
                smiles, sequence, interaction = data.strip().split()

                # count positive samples
                positive += int(interaction)
                words = self.split_sequence(sequence, ngram)

                molecules.append(smiles)
                labels.append(torch.LongTensor([int(interaction)]))
                proteins.append(torch.LongTensor(words))
                protein_lens.append(torch.LongTensor([len(words)]))

                # mol = MolFromSmiles(smiles)
                # if mol == None:
                #     continue
                # data = mol_to_graph_data_dic(mol)
                # data.y = torch.LongTensor([int(interaction)])
                # words = split_sequence(sequence, ngram)
                # data.protein = torch.LongTensor(words)
                # data.pr_len = torch.LongTensor([len(words)])
                # DATALIST.append(data)

            assert len(molecules) == len(proteins) == len(labels)

            lines = list(list(t) for t in zip(molecules, proteins, protein_lens, labels))

            data = [
                MoleculeProteinDatapoint(
                    line=line,
                    args=args,
                    features=features_data if features_data is not None else None,
                    use_compound_names=use_compound_names
                ) for i, line in tqdm(enumerate(lines), total=len(lines), disable=True)
            ]

            weights = [len(data) / (len(data) - positive), len(data) / positive]
            print(weights)

            # Filter out invalid SMILES
            if skip_invalid_smiles:
                original_data_len = len(data)
                data = self.filter_invalid_smiles(data)

                if len(data) < original_data_len:
                    print(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

        return data

    def filter_invalid_smiles(self, data: MoleculeDataset) -> MoleculeDataset:
        """
        Filters out invalid SMILES.

        :param data: A MoleculeDataset.
        :return: A MoleculeDataset with only valid molecules.
        """
        datapoint_list = []
        for idx, datapoint in enumerate(data):
            if datapoint.smiles == '':
                print(f'invalid smiles {idx}: {datapoint.smiles}')
                continue
            mol = Chem.MolFromSmiles(datapoint.smiles)
            if mol == None:
                print(f'invalid smiles1 {idx}: {datapoint.smiles}')
                continue
            if mol.GetNumHeavyAtoms() == 0:
                print(f'invalid heavy {idx}')
                continue

            datapoint_list.append(datapoint)
        return datapoint_list
        # return MoleculeDataset(datapoint_list)

    def split_sequence(self, sequence, ngram):
        sequence = '_' + sequence + '='
        words = [self.word_dict[sequence[i:i + ngram]]
                 for i in range(len(sequence) - ngram + 1)]
        return np.array(words)







# def split_sequence(sequence, ngram):
#     sequence = '_' + sequence + '='
#     words = [self.word_dict[sequence[i:i + ngram]]
#              for i in range(len(sequence) - ngram + 1)]
#     return np.array(words)
#
# def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
#     """
#     Filters out invalid SMILES.
#
#     :param data: A MoleculeDataset.
#     :return: A MoleculeDataset with only valid molecules.
#     """
#     datapoint_list = []
#     for idx, datapoint in enumerate(data):
#         if datapoint.smiles == '':
#             print(f'invalid smiles {idx}: {datapoint.smiles}')
#             continue
#         mol = Chem.MolFromSmiles(datapoint.smiles)
#         if mol == None:
#             print(f'invalid smiles1 {idx}: {datapoint.smiles}')
#             continue
#         if mol.GetNumHeavyAtoms() == 0:
#             print(f'invalid heavy {idx}')
#             continue
#
#         datapoint_list.append(datapoint)
#     return MoleculeDataset(datapoint_list)
#
# def get_dti_data(args, root, dataset_name,
#                  skip_invalid_smiles: bool = True,
#                  features_path: List[str] = None,
#                  max_data_size: int = None,
#                  use_compound_names: bool = None
#                  ):
#     print('Loading data')
#     if args is not None:
#         # Prefer explicit function arguments but default to args if not provided
#         features_path = features_path if features_path is not None else args.features_path
#         max_data_size = max_data_size if max_data_size is not None else args.max_data_size
#         use_compound_names = use_compound_names if use_compound_names is not None else args.use_compound_names
#     else:
#         use_compound_names = False
#
#     max_data_size = max_data_size or float('inf')
#     # Load features
#     if features_path is not None:
#         features_data = []
#         # for feat_path in features_path:
#         #     features_data.append(load_features(feat_path))  # each is num_data x num_features
#         feature = load_features(features_path)
#         features_data.append(feature)
#
#         features_data = np.concatenate(features_data, axis=1)
#         args.features_dim = len(features_data[0])
#     else:
#         features_data = None
#         if args is not None:
#             args.features_dim = 0
#
#     if dataset_name == 'human':
#         with open(root, 'r') as f:
#             data_list = f.read().strip().split('\n')
#         data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
#         N = len(data_list)
#
#         ngram = 3
#         positive = 0
#         molecules = []
#         proteins = []
#         protein_lens = []
#         labels = []
#         for no, data in enumerate(tqdm(data_list)):
#             smiles, sequence, interaction = data.strip().split()
#
#             # count positive samples
#             positive += int(interaction)
#             words = split_sequence(sequence, ngram)
#
#             molecules.append(smiles)
#             labels.append(torch.LongTensor([int(interaction)]))
#             proteins.append(torch.LongTensor(words))
#             protein_lens.append(torch.LongTensor([len(words)]))
#
#             # mol = MolFromSmiles(smiles)
#             # if mol == None:
#             #     continue
#             # data = mol_to_graph_data_dic(mol)
#             # data.y = torch.LongTensor([int(interaction)])
#             # words = split_sequence(sequence, ngram)
#             # data.protein = torch.LongTensor(words)
#             # data.pr_len = torch.LongTensor([len(words)])
#             # DATALIST.append(data)
#
#
#
#         assert len(molecules) == len(proteins) == len(labels)
#
#         lines = list(list(t) for t in zip(molecules, proteins, protein_lens, labels))
#
#         data = [
#             MoleculeProteinDatapoint(
#                 line=line,
#                 args=args,
#                 features=features_data if features_data is not None else None,
#                 use_compound_names=use_compound_names
#             ) for i, line in tqdm(enumerate(lines), total=len(lines), disable=True)
#         ]
#
#         weights = [len(data) / (len(data) - positive), len(data) / positive]
#         print(weights)
#
#         # Filter out invalid SMILES
#         if skip_invalid_smiles:
#             original_data_len = len(data)
#             data = filter_invalid_smiles(data)
#
#             if len(data) < original_data_len:
#                 print(f'Warning: {original_data_len - len(data)} SMILES are invalid.')
#
#     return data








class TestDataset(torch.utils.data.Dataset):
    def __init__(self,file_path):
        self.path = file_path
        self.datalist = []
        with open('data/downstream/3ngram_vocab', 'r') as f:
            word_dic = f.read().split('\n')
            if word_dic[-1] == '':
                word_dic.pop()

        # word_dic = ['pad'] + ['other{}'.format(n) for n in range(10)] + word_dic
        self.word_dict = {}
        for i, item in enumerate(word_dic):
            self.word_dict[item] = i
        self.n_word = len(self.word_dict)

        self.process()

    def __getitem__(self, idx):
        data = self.datalist[idx]  # attention: must self.indices()[idx]
        return data

    def __len__(self):
        return len(self.datalist)
    def split_sequence(self,sequence, ngram):
        sequence = '_' + sequence + '='
        words = [self.word_dict[sequence[i:i + ngram]]
                 for i in range(len(sequence) - ngram + 1)]
        return np.array(words)
    def process(self):

        with open(self.path,'r') as f:
            data_list = f.read().split('\n')
            if not data_list[-1]:
                data_list.pop()

        self.ngram = 3
        positive = 0
        DATALIST = []
        for no, data in enumerate(tqdm(data_list)):
            smiles, sequence, interaction = data.strip().split()

            # count positive samples
            positive+=int(interaction)

            mol = MolFromSmiles(smiles)
            if mol==None:
                continue
            data = mol_to_graph_data_dic(mol)
            data.y = torch.LongTensor([int(interaction)])
            words = self.split_sequence(sequence, self.ngram)
            data.protein = torch.LongTensor(words)
            data.pr_len = torch.LongTensor([len(words)])
            DATALIST.append(data)
        self.datalist=DATALIST



class MultiDataset(InMemoryDataset):

    def __init__(self, root, dataset, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        with open('../data/downstream/3ngram_vocab', 'r') as f:
            word_dic = f.read().split('\n')
            if word_dic[-1] == '':
                word_dic.pop()
        # word_dic=['pad']+['other{}'.format(n) for n in range(10)]+word_dic
        self.word_dict = {}
        for i, item in enumerate(word_dic):
            self.word_dict[item] = i
        self.n_word = len(self.word_dict)
        super(MultiDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.txt']

    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]

    def download(self):
        pass

    def split_sequence(self,sequence, ngram):
        sequence = '_' + sequence + '='
        words = [self.word_dict[sequence[i:i + ngram]]
                 for i in range(len(sequence) - ngram + 1)]
        return np.array(words)

    def dump_dictionary(self,dictionary, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(dictionary), f)

    def process(self):

        if self.dataset=='bindingdb':
            from preprocess_dataset import process_BindingDB

        else:
            with open(self.raw_paths[0], 'r') as f:
                data_list = f.read().strip().split('\n')
            data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
            N = len(data_list)

            self.ngram = 3
            positive = 0
            DATALIST = []
            for no, data in enumerate(tqdm(data_list)):
                smiles, sequence, interaction = data.strip().split()

                # count positive samples
                positive+=int(interaction)

                mol = MolFromSmiles(smiles)
                if mol==None:
                    continue
                data = mol_to_graph_data_dic(mol)
                data.y = torch.LongTensor([int(interaction)])
                words = self.split_sequence(sequence, self.ngram)
                data.protein = torch.LongTensor(words)
                data.pr_len = torch.LongTensor([len(words)])
                DATALIST.append(data)

            weights = [len(DATALIST)/(len(DATALIST)-positive),len(DATALIST)/positive]
            print(weights)

        print(len(self.word_dict))

        if self.pre_filter is not None:
            DATALIST = [data for data in DATALIST if self.pre_filter(data)]
        if self.pre_transform is not None:
            DATALIST= [self.pre_transform(data) for data in DATALIST]

        data, slices = self.collate(DATALIST)
        torch.save((data, slices), self.processed_paths[0])

def load_dataset_random(args, path, dataset, seed, tasks=None):
    # save_path = path + 'processed/train_valid_test_{}_seed_{}.ckpt'.format(dataset, seed)
    # if os.path.isfile(save_path):
    #     trn, val, test = torch.load(save_path)
    #     return trn, val, test

    data = Get_DTI_Data(args=args, root=path, dataset_name=dataset)
    dataset = MoleculeDataset(data.data)
    n_word = data.n_word

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    dataset.shuffle()
    trn, val, test = dataset[:train_size], \
                     dataset[train_size:(train_size + val_size)], \
                     dataset[(train_size + val_size):]
    # trn.weights = 'regression task has no class weights!'
    #
    # torch.save([trn, val, test], save_path)
    return MoleculeDataset(trn), MoleculeDataset(val), MoleculeDataset(test), n_word

if __name__=='__main__':

    # from args import *
    import torch,random,os
    # import numpy as np
    #
    # def seed_set(seed=1029):
    #     random.seed(seed)
    #     os.environ['PYTHONHASHSEED'] = str(seed)
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #
    #
    # if args.dataset=='human':
    #     raw = os.path.join(args.data,args.dataset)+'/raw/data.txt'
    #
    #     dataset = MultiDataset(os.path.join(args.data, args.dataset), args.dataset)
    #
    #     i=0
    #     for seed in [90,34,33]:
    #         seed_set(seed)
    #         perm = torch.randperm(len(dataset)).numpy()
    #
    #         with open(raw, 'r') as f:
    #             data_list = f.read().strip().split('\n')
    #         data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    #         N = len(data_list)
    #
    #
    #         data_list=np.array(data_list)[perm]
    #
    #         train_size = int(0.8 * len(dataset))
    #         val_size = int(0.1 * len(dataset))
    #         test_size = len(dataset) - train_size - val_size
    #
    #         trn = data_list[:train_size]
    #         val = data_list[train_size:(train_size+val_size)]
    #         test = data_list[(train_size+val_size):]
    #
    #         save_dir = 'finetuned_model/DTI/{}/fold_{}/'.format(args.dataset,i)
    #         print(save_dir)
    #         with open('{}/train.txt'.format(save_dir),'w') as f:
    #             for item in trn:
    #                 f.write(item)
    #                 f.write('\n')
    #         with open('{}/valid.txt'.format(save_dir),'w') as f:
    #             for item in val:
    #                 f.write(item)
    #                 f.write('\n')
    #         with open('{}/test.txt'.format(save_dir),'w') as f:
    #             for item in test:
    #                 f.write(item)
    #                 f.write('\n')
    #         i+=1
    # if args.dataset == 'celegans':
    #     dataset = MultiDataset(os.path.join(args.data, args.dataset), args.dataset)
    #     for seed in [90,88,33]:
    #         seed_set(seed)
    #         perm = torch.randperm(len(dataset)).numpy()
    #         raw = os.path.join(args.data, args.dataset) + '/raw/data.txt'
    #         with open(raw, 'r') as f:
    #             data_list = f.read().strip().split('\n')
    #         data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    #         N = len(data_list)
    #
    #         data_list=np.array(data_list)[perm]
    #
    #         train_size = int(0.8 * len(dataset))
    #         val_size = int(0.1 * len(dataset))
    #         test_size = len(dataset) - train_size - val_size
    #
    #         trn = data_list[:train_size]
    #         val = data_list[train_size:(train_size+val_size)]
    #         test = data_list[(train_size+val_size):]
    #
    #         save_dir = 'finetuned_model/DTI/{}/seed_{}/'.format(args.dataset,seed)
    #         print(save_dir)
    #         with open('{}/train.txt'.format(save_dir),'w') as f:
    #             for item in trn:
    #                 f.write(item)
    #                 f.write('\n')
    #         with open('{}/valid.txt'.format(save_dir),'w') as f:
    #             for item in val:
    #                 f.write(item)
    #                 f.write('\n')
    #         with open('{}/test.txt'.format(save_dir),'w') as f:
    #             for item in test:
    #                 f.write(item)
    #                 f.write('\n')
    # pyg_dataset = CancerDataset('dataset/cancer', 'cancer')
    # pyg_dataset = MultiDataset2('dataset/bindingdb','bindingdb',y = 'kd')
    # train_size = int(0.8 * len(pyg_dataset))
    # val_size = int(0.1 * len(pyg_dataset))
    # test_size = len(pyg_dataset) - train_size - val_size
    # pyg_dataset = pyg_dataset.shuffle()
    # trn, val, test = pyg_dataset[:train_size], \
    #                  pyg_dataset[train_size:(train_size + val_size)], \
    #                  pyg_dataset[(train_size + val_size):]
    # print(val,test)
    # save_path = 'dataset/'+ 'processed/train_valid_test_seed.ckpt'
    # torch.save([trn, val, test], save_path)
    #
    # print(trn[0])