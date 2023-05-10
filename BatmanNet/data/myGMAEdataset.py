"""
The dataset used in training GROVER.
"""
import math
import os
import csv
import pickle
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from rdkit import Chem
import random
from operator import itemgetter

import BatmanNet.util.utils as feautils
from BatmanNet.data.molgraph import mol2graph, ELEM_LIST
from BatmanNet.data.moldataset import MoleculeDatapoint
from BatmanNet.data.task_labels import atom_to_vocab, bond_to_vocab


def get_data_processed(data_path,args):
    """
    Load data from the data_path.
    :param data_path: the data_path.
    :param logger: the logger.
    :return:
    """
    print('Read processed data')
    data_path = os.path.join(data_path, 'batch_size={}_mask_ratio={}'.format(args.batch_size, args.mask_ratio))
    with open(os.path.join(data_path, 'train_mol_graph.pkl'), 'rb') as f:
        train_data = pickle.load(f)
    with open(os.path.join(data_path, 'test_mol_graph.pkl'), 'rb') as f:
        test_data = pickle.load(f)
    print('Read processed data finish')
    return MolDataset(train_data), MolDataset(test_data)


def split_data(data,
               args,
               sizes=(0.8, 0.1, 0.1),
               logger=None):
    """
    Split data with given train/validation/test ratio.
    :param data:
    :param split_type:
    :param sizes:
    :param seed:
    :param logger:
    :return:
    """
    assert len(sizes) == 3 and sum(sizes) == 1

    train_size = int(sizes[0] * len(data))
    train_val_size = int((sizes[0] + sizes[1]) * len(data))

    train = data[:train_size]
    val = data[train_size:train_val_size]
    test = data[train_val_size:]
    return MolDataset(train), MolDataset(val), MolDataset(test)

class MolDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.len = len(data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return self.data[idx]

class myGMAE_Collator(object):
    def __init__(self, shared_dict, args):
        self.args = args
        self.mask_ratio = args.mask_ratio
        self.shared_dict = shared_dict

    def node_random_masking(self, graph_batch, mask_radio):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = graph_batch

        mask_idx_list = []
        remain_idx_list = []
        new_ascope = []
        masked_natoms = 1
        for a in a_scope:
            start_inx, mol_node_num = a
            # mol_atoms = f_atoms[start_inx: start_inx+mol_len]
            n_mask = math.ceil(mol_node_num * mask_radio)
            single_mask_idx = np.random.permutation(mol_node_num.item())[:n_mask]  # 对0-mol_len之间的数随机排序
            mask_idx = [(m + start_inx).item() for m in single_mask_idx]
            mask_idx.sort()
            # mask_idx_list.append(mask_idx)
            mask_idx_list.extend(mask_idx)

            remain_mol_node_num = mol_node_num.item() - n_mask
            new_ascope.append((masked_natoms, remain_mol_node_num))
            masked_natoms += remain_mol_node_num

        for idx in range(len(f_atoms)):
            if idx not in mask_idx_list:
                remain_idx_list.append(idx)

        assert len(mask_idx_list) + len(remain_idx_list) == len(f_atoms)
        assert sorted(mask_idx_list + remain_idx_list) == [i for i in range(len(f_atoms))]

        getter_mask = itemgetter(mask_idx_list)
        getter_remain = itemgetter(remain_idx_list)

        mask_fatoms = getter_mask(f_atoms)
        remain_fatoms = getter_remain(f_atoms)

        remain_a2a = getter_remain(a2a)
        remain_a2b = getter_remain(a2b)

        remain_a2a = remain_a2a.tolist()
        new_remain_a2a = []
        for rea in remain_a2a:
            for idx, a in enumerate(rea):
                if a in mask_idx_list:
                    rea[idx] = int(0)
                else:
                    rea[idx] = remain_idx_list.index(a)
            new_remain_a2a.append(rea)

        remain_b2a = b2a.tolist()
        for idx, a in enumerate(remain_b2a):
            if a in mask_idx_list:
                remain_b2a[idx] = 0
            else:
                remain_b2a[idx] = remain_idx_list.index(a)

        mask_fatoms = torch.FloatTensor(mask_fatoms)
        remain_fatoms = torch.FloatTensor(remain_fatoms)
        new_remain_a2a = torch.LongTensor(new_remain_a2a).long()
        remain_a2b = torch.LongTensor(remain_a2b)
        remain_b2a = torch.LongTensor(remain_b2a)
        new_ascope = torch.LongTensor(new_ascope)

        node_masked_batch = (mask_idx_list, remain_idx_list, mask_fatoms, remain_fatoms, new_remain_a2a, remain_a2b,
                             remain_b2a, new_ascope)
        return node_masked_batch

    def edge_random_masking(self, graph_batch, mask_radio):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, a2a = graph_batch

        mask_idx_list = []
        remain_idx_list = []
        new_bscope = []
        masked_nbonds = 1
        for b in b_scope:
            start_inx, mol_bond_num = b
            n_mask = math.ceil(mol_bond_num * mask_radio)
            single_mask_idx = np.random.permutation(mol_bond_num.item())[:n_mask]  # 对0-mol_len之间的数随机排序
            mask_idx = [(m + start_inx).item() for m in single_mask_idx]
            mask_idx.sort()
            # mask_idx_list.append(mask_idx)
            mask_idx_list.extend(mask_idx)

            remain_mol_bond_num = mol_bond_num.item() - n_mask
            new_bscope.append((masked_nbonds, remain_mol_bond_num))
            masked_nbonds += remain_mol_bond_num

        for idx in range(len(f_bonds)):
            if idx not in mask_idx_list:
                remain_idx_list.append(idx)

        assert len(mask_idx_list) + len(remain_idx_list) == len(f_bonds)
        assert sorted(mask_idx_list + remain_idx_list) == [i for i in range(len(f_bonds))]

        getter_mask = itemgetter(mask_idx_list)
        getter_remain = itemgetter(remain_idx_list)

        mask_fbonds = getter_mask(f_bonds)
        remain_fbonds = getter_remain(f_bonds)
        remain_b2a = getter_remain(b2a)
        remain_b2revb = getter_remain(b2revb)

        remain_b2revb = remain_b2revb.tolist()
        for idx, b in enumerate(remain_b2revb):
            if b in mask_idx_list:
                remain_b2revb[idx] = 0
            else:
                remain_b2revb[idx] = remain_idx_list.index(b)

        remain_a2b = a2b.tolist()
        new_remain_a2b = []
        for rea in remain_a2b:
            for idx, b in enumerate(rea):
                if b in mask_idx_list:
                    rea[idx] = 0
                else:
                    rea[idx] = remain_idx_list.index(b)
            new_remain_a2b.append(rea)

        mask_fbonds = torch.FloatTensor(mask_fbonds)
        remain_fbonds = torch.FloatTensor(remain_fbonds)
        remain_b2a = torch.LongTensor(remain_b2a).long()
        new_remain_a2b = torch.LongTensor(new_remain_a2b)
        remain_b2revb = torch.LongTensor(remain_b2revb)
        new_bscope = torch.LongTensor(new_bscope)

        edge_masked_batch = (mask_idx_list, remain_idx_list, mask_fbonds, remain_fbonds, remain_b2a, new_remain_a2b,
                             remain_b2revb, new_bscope)
        return edge_masked_batch

    def get_mask_target_features(self, node_masked_batch, edge_masked_batch):
        mask_node_features, mask_edge_features = node_masked_batch[2], edge_masked_batch[2]
        num_atom_type = len(ELEM_LIST)

        mask_node_atomtype = mask_node_features[:, : num_atom_type].argmax(1)
        mask_node_degree = mask_node_features[:, num_atom_type: num_atom_type + 6].argmax(1)
        mask_node_charge = mask_node_features[:, num_atom_type + 6: num_atom_type + 11].argmax(1)
        mask_node_chirality = mask_node_features[:, num_atom_type + 11: num_atom_type + 15].argmax(1)
        mask_node_is_aromatic = mask_node_features[:, num_atom_type + 15: num_atom_type + 16].argmax(1)

        mask_edge_node_atomtype = mask_edge_features[:, : num_atom_type].argmax(1)
        mask_edge_node_degree = mask_edge_features[:, num_atom_type: num_atom_type + 6].argmax(1)
        mask_edge_node_charge = mask_edge_features[:, num_atom_type + 6: num_atom_type + 11].argmax(1)
        mask_edge_node_chirality = mask_edge_features[:, num_atom_type + 11: num_atom_type + 15].argmax(1)
        mask_edge_node_is_aromatic = mask_edge_features[:, num_atom_type + 15: num_atom_type + 16].argmax(1)
        mask_edge_bond_type = mask_edge_features[:, num_atom_type + 16: num_atom_type + 21].argmax(1)
        mask_edge_stereo_confi = mask_edge_features[:, num_atom_type + 21: num_atom_type + 27].argmax(1)

        # CrossEntropyLoss的target必须为long(),不能是one-hot格式，
        # BCEWithLogitsLoss()的target需要和pred的数据类型一致，维度也需要一致
        mask_node_target = (mask_node_atomtype.long(), mask_node_degree.long(),
                            mask_node_charge.long(), mask_node_chirality.long(),
                            mask_node_is_aromatic.float())
        mask_edge_target = (mask_edge_node_atomtype.long(), mask_edge_node_degree.long(), mask_edge_node_charge.long(),
                            mask_edge_node_chirality.long(), mask_edge_node_is_aromatic.float(), mask_edge_bond_type.long(),
                            mask_edge_stereo_confi.long())
        return mask_node_target, mask_edge_target


    def __call__(self, batch):
        # smiles_batch = [d for d in batch]
        smiles_batch = batch
        mask_radio = self.mask_ratio
        graph_batch = mol2graph(smiles_batch, self.shared_dict, self.args).get_components()

        node_masked_batch = self.node_random_masking(graph_batch, mask_radio)
        edge_masked_batch = self.edge_random_masking(graph_batch, mask_radio)
        mask_node_target, mask_edge_target = self.get_mask_target_features(node_masked_batch, edge_masked_batch)

        # atom_vocab_label = torch.Tensor(self.atom_random_mask(smiles_batch, self.args.mask_percent)).long()
        # bond_vocab_label = torch.Tensor(self.bond_random_mask(smiles_batch)).long()
        # fgroup_label = torch.Tensor([d.features for d in batch]).float()
        # may be some mask here
        # res = {"graph_input": batchgraph,
        #        "targets": {"av_task": atom_vocab_label,
        #                    "bv_task": bond_vocab_label,
        #                    "fg_task": fgroup_label}
        #        }
        return (graph_batch, node_masked_batch, edge_masked_batch, mask_node_target, mask_edge_target)

    # def atom_random_mask(self, smiles_batch, percent):
    #     """
    #     Perform the random mask operation on atoms.
    #     :param smiles_batch:
    #     :return: The corresponding atom labels.
    #     """
    #     for smi in smiles_batch:
    #         mol = Chem.MolFromSmiles(smi)
    #         n_mask = math.ceil(mol.GetNumAtoms() * percent)
    #         perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]

        # There is a zero padding.
        # vocab_label = [0]
        # percent = 0.15
        # for smi in smiles_batch:
        #     mol = Chem.MolFromSmiles(smi)
        #     mlabel = [0] * mol.GetNumAtoms()
        #     n_mask = math.ceil(mol.GetNumAtoms() * percent)
        #     perm = np.random.permutation(mol.GetNumAtoms())[:n_mask]
        #     for p in perm:
        #         atom = mol.GetAtomWithIdx(int(p))
        #         mlabel[p] = self.atom_vocab.stoi.get(atom_to_vocab(mol, atom), self.atom_vocab.other_index)
        #
        #     vocab_label.extend(mlabel)
        # return vocab_label

    # def bond_random_mask(self, smiles_batch):
    #     """
    #     Perform the random mask operaiion on bonds.
    #     :param smiles_batch:
    #     :return: The corresponding bond labels.
    #     """
    #     # There is a zero padding.
    #     vocab_label = [0]
    #     percent = 0.15
    #     for smi in smiles_batch:
    #         mol = Chem.MolFromSmiles(smi)
    #         nm_atoms = mol.GetNumAtoms()
    #         nm_bonds = mol.GetNumBonds()
    #         mlabel = []
    #         n_mask = math.ceil(nm_bonds * percent)
    #         perm = np.random.permutation(nm_bonds)[:n_mask]
    #         virtual_bond_id = 0
    #         for a1 in range(nm_atoms):
    #             for a2 in range(a1 + 1, nm_atoms):
    #                 bond = mol.GetBondBetweenAtoms(a1, a2)
    #
    #                 if bond is None:
    #                     continue
    #                 if virtual_bond_id in perm:
    #                     label = self.bond_vocab.stoi.get(bond_to_vocab(mol, bond), self.bond_vocab.other_index)
    #                     mlabel.extend([label])
    #                 else:
    #                     mlabel.extend([0])
    #
    #                 virtual_bond_id += 1
    #         # todo: might need to consider bond_drop_rate
    #         # todo: double check reverse bond
    #         vocab_label.extend(mlabel)
    #     return vocab_label



# class BatchDatapoint:
#     def __init__(self,
#                  smiles_file,
#                  feature_file,
#                  n_samples,
#                  ):
#         self.smiles_file = smiles_file
#         self.feature_file = feature_file
#         # deal with the last batch graph numbers.
#         self.n_samples = n_samples
#         self.datapoints = None
#
#     def load_datapoints(self):
#         features = self.load_feature()
#         self.datapoints = []
#
#         with open(self.smiles_file) as f:
#             reader = csv.reader(f)
#             next(reader)
#             for i, line in enumerate(reader):
#                 # line = line[0]
#                 d = MoleculeDatapoint(line=line,
#                                       features=features[i])
#                 self.datapoints.append(d)
#
#         assert len(self.datapoints) == self.n_samples
#
#     def load_feature(self):
#         return feautils.load_features(self.feature_file)
#
#     def shuffle(self):
#         pass
#
#     def clean_cache(self):
#         del self.datapoints
#         self.datapoints = None
#
#     def __len__(self):
#         return self.n_samples
#
#     def __getitem__(self, idx):
#         assert self.datapoints is not None
#         return self.datapoints[idx]
#
#     def is_loaded(self):
#         return self.datapoints is not None



# class BatchMolDataset(Dataset):
#     def __init__(self, data, batch_size, shuffle=True):
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         if len(data) == 0:
#             self.batches = []
#         else:
#             if self.shuffle:
#                 random.shuffle(data)
#             batches = [data[i: i + self.batch_size] for i in range(0, len(data), self.batch_size)]
#             if len(batches[-1]) < self.batch_size:
#                 batches.pop()
#             self.batches = batches
#
#     def __len__(self):
#         return len(self.batches)
#
#     def __getitem__(self, idx):
#         return self.batches[idx]
        # return tensorize(self.batches[idx])


    # def __init__(self, data: List[BatchDatapoint],
    #              graph_per_file=None):
    #     self.data = data
    #
    #     self.len = 0
    #     for d in self.data:
    #         self.len += len(d)
    #     if graph_per_file is not None:
    #         self.sample_per_file = graph_per_file
    #     else:
    #         self.sample_per_file = len(self.data[0]) if len(self.data) != 0 else None
    #
    # def shuffle(self, seed: int = None):
    #     pass
    #
    # def clean_cache(self):
    #     for d in self.data:
    #         d.clean_cache()
    #
    # def __len__(self) -> int:
    #     return self.len
    #
    # def __getitem__(self, idx) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
    #     # print(idx)
    #     dp_idx = int(idx / self.sample_per_file)
    #     real_idx = idx % self.sample_per_file
    #     return self.data[dp_idx][real_idx]
    #
    # def load_data(self, idx):
    #     dp_idx = int(idx / self.sample_per_file)
    #     if not self.data[dp_idx].is_loaded():
    #         self.data[dp_idx].load_datapoints()
    #
    # def count_loaded_datapoints(self):
    #     res = 0
    #     for d in self.data:
    #         if d.is_loaded():
    #             res += 1
    #     return res




