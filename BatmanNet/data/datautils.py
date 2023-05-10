import torch
import numpy as np
import math
from operator import itemgetter

from BatmanNet.data.molgraph import ELEM_LIST


def split_data(data, sizes=(0.8, 0.1, 0.1)):
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
    return train, val, test

def node_random_masking(graph_batch, mask_radio):
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

def edge_random_masking(graph_batch, mask_radio):
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

def get_mask_target_features(node_masked_batch, edge_masked_batch):
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