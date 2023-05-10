"""
The GROVER models for pretraining, finetuning and fingerprint generating.
"""
from argparse import Namespace
from typing import List, Dict, Callable

import numpy as np
import math
import torch
from torch import nn as nn
from operator import itemgetter
import time
from BatmanNet.data.molgraph import get_atom_fdim, get_bond_fdim, ELEM_LIST
from BatmanNet.model.layers import Readout, GTransEncoder, GTransDecoder
from BatmanNet.util.nn_utils import get_activation_function


class BatmanNet_model(nn.Module):
    """
    The GROVER Embedding class. It contains the GTransEncoder.
    This GTransEncoder can be replaced by any validate encoders.
    """

    def __init__(self, args: Namespace):
        """
        Initialize MYGMAE_model class.
        :param args:
        """
        super(BatmanNet_model, self).__init__()

        self.mask_ratio = args.mask_ratio
        self.hidden_size = args.hidden_size
        # self.embedding_output_type = args.embedding_output_type
        edge_dim = get_bond_fdim() + get_atom_fdim()
        node_dim = get_atom_fdim()

        # self.pos_embed = nn.Parameter(torch.zeros(1, num_atoms + 1, embed_dim), requires_grad=False)
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.encoders = GTransEncoder(args,
                                      hidden_size=args.hidden_size,
                                      edge_fdim=edge_dim,
                                      node_fdim=node_dim,
                                      dropout=args.dropout,
                                      activation=args.activation,
                                      num_mt_block=args.num_enc_mt_block,
                                      num_attn_head=args.num_attn_head,
                                      # atom_emb_output=self.embedding_output_type,
                                      bias=args.bias,
                                      cuda=args.cuda,
                                      res_connection=False)

        self.decoders = GTransDecoder(args,
                                      hidden_size=args.hidden_size,
                                      edge_fdim=edge_dim,
                                      node_fdim=node_dim,
                                      dropout=args.dropout,
                                      activation=args.activation,
                                      num_mt_block=args.num_dec_mt_block,
                                      num_attn_head=args.num_attn_head,
                                      # atom_emb_output=self.embedding_output_type,
                                      bias=args.bias,
                                      cuda=args.cuda,
                                      res_connection=False)


    def get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        # return torch.FloatTensor(sinusoid_table).unsequeeze(0)
        return torch.FloatTensor(sinusoid_table)


    def get_pos_embed(self, graph_batch, node_masked_batch, edge_masked_batch, d_hid):
        f_atoms, f_bonds, _, _, _, _, _, _ = graph_batch
        node_mask_idx_list, node_remain_idx_list, _, _, _, _, _, _ = node_masked_batch
        edge_mask_idx_list, edge_remain_idx_list, _, _, _, _, _, _ = edge_masked_batch
        # All start with zero padding so that indexing with zero padding returns zeros
        atoms_pos_embed = [[0] * d_hid]  # atom features
        edges_pos_embed = [[0] * d_hid]  # combined atom/bond features
        num_atoms = len(f_atoms) - 1
        num_edges = len(f_bonds) - 1
        atoms_pos_embed.extend(self.get_sinusoid_encoding_table(num_atoms, d_hid))
        edges_pos_embed.extend(self.get_sinusoid_encoding_table(num_edges, d_hid))
        atoms_pos_embed = torch.FloatTensor(atoms_pos_embed)
        bonds_pos_embed = torch.FloatTensor(edges_pos_embed)

        getter_node_mask = itemgetter(node_mask_idx_list)
        getter_node_remain = itemgetter(node_remain_idx_list)
        getter_edge_mask = itemgetter(edge_mask_idx_list)
        getter_edge_remain = itemgetter(edge_remain_idx_list)

        mask_atom_pos_embed = getter_node_mask(atoms_pos_embed)
        remain_atom_pos_embed = getter_node_remain(atoms_pos_embed)
        mask_edge_pos_embed = getter_edge_mask(bonds_pos_embed)
        remain_edge_pos_embed = getter_edge_remain(bonds_pos_embed)

        return mask_atom_pos_embed, remain_atom_pos_embed, mask_edge_pos_embed, remain_edge_pos_embed


    def feature_reordering(self, remain_atom_pos_embed, remain_edge_pos_embed, mask_atom_pos_embed, mask_edge_pos_embed,
                           node_mask_idx_list, node_remain_idx_list, edge_mask_idx_list, edge_remain_idx_list,
                           node_embeddings, edge_embeddings):
        """特征重排序"""
        remain_atom_pos_embed, remain_edge_pos_embed = remain_atom_pos_embed.cuda(), remain_edge_pos_embed.cuda()
        mask_atom_pos_embed, mask_edge_pos_embed = mask_atom_pos_embed.cuda(), mask_edge_pos_embed.cuda()
        node_embeddings = node_embeddings + remain_atom_pos_embed
        edge_embeddings = edge_embeddings + remain_edge_pos_embed
        # 节点特征重排序
        node_dec_input = torch.vstack((node_embeddings, mask_atom_pos_embed))
        node_idx_list = np.hstack((node_remain_idx_list, node_mask_idx_list))
        node_idx_list_sorted_idx = np.argsort(node_idx_list)
        node_dec_input = node_dec_input[node_idx_list_sorted_idx]

        # 边特征重排序
        edge_dec_input = torch.vstack((edge_embeddings, mask_edge_pos_embed))
        edge_idx_list = np.hstack((edge_remain_idx_list, edge_mask_idx_list))
        edge_idx_list_sorted_idx = np.argsort(edge_idx_list)
        edge_dec_input = edge_dec_input[edge_idx_list_sorted_idx]

        return node_dec_input, edge_dec_input


    def get_mask_predict_features(self, node_features, edge_features, node_mask_idx_list, edge_mask_idx_list):
        getter_node_mask = itemgetter(node_mask_idx_list)
        getter_edge_mask = itemgetter(edge_mask_idx_list)

        mask_node_features = getter_node_mask(node_features)
        mask_edge_features = getter_edge_mask(edge_features)

        num_atom_type = len(ELEM_LIST)

        mask_node_atomtype = mask_node_features[:, : num_atom_type]
        mask_node_degree = mask_node_features[:, num_atom_type: num_atom_type + 6]
        mask_node_charge = mask_node_features[:, num_atom_type + 6: num_atom_type + 11]
        mask_node_chirality = mask_node_features[:, num_atom_type + 11: num_atom_type + 15]
        mask_node_is_aromatic = mask_node_features[:, num_atom_type + 15: num_atom_type + 16]

        mask_edge_node_atomtype = mask_edge_features[:, : num_atom_type]
        mask_edge_node_degree = mask_edge_features[:, num_atom_type: num_atom_type + 6]
        mask_edge_node_charge = mask_edge_features[:, num_atom_type + 6: num_atom_type + 11]
        mask_edge_node_chirality = mask_edge_features[:, num_atom_type + 11: num_atom_type + 15]
        mask_edge_node_is_aromatic = mask_edge_features[:, num_atom_type + 15: num_atom_type + 16]
        mask_edge_bond_type = mask_edge_features[:, num_atom_type + 16: num_atom_type + 21]
        mask_edge_stereo_confi = mask_edge_features[:, num_atom_type + 21: num_atom_type + 27]

        mask_node_pre = (mask_node_atomtype, mask_node_degree, mask_node_charge, mask_node_chirality,
                         mask_node_is_aromatic)
        mask_edge_pre = (mask_edge_node_atomtype, mask_edge_node_degree, mask_edge_node_charge, mask_edge_node_chirality
                         , mask_edge_node_is_aromatic, mask_edge_bond_type, mask_edge_stereo_confi)
        return mask_node_pre, mask_edge_pre


    def forward(self, graph_batch, node_masked_batch, edge_masked_batch):
        """
        The forward function takes graph_batch as input and output a dict. The content of the dict is decided by
        self.embedding_output_type.

        :param graph_batch: the input graph batch generated by MolCollator.
        :return: a dict containing the embedding results.
        """
        # mask_radio = self.mask_ratio
        hidden_size = self.hidden_size

        # node_masked_batch = self.node_random_masking(graph_batch, mask_radio)
        # edge_masked_batch = self.edge_random_masking(graph_batch, mask_radio)

        node_mask_idx_list = node_masked_batch[0]
        node_remain_idx_list = node_masked_batch[1]
        edge_mask_idx_list = edge_masked_batch[0]
        edge_remain_idx_list = edge_masked_batch[1]

        mask_atom_pos_embed, remain_atom_pos_embed, mask_edge_pos_embed, remain_edge_pos_embed = \
            self.get_pos_embed(graph_batch, node_masked_batch, edge_masked_batch, hidden_size)

        node_embeddings, edge_embeddings = self.encoders(graph_batch, node_masked_batch, edge_masked_batch,
                                                         remain_atom_pos_embed, remain_edge_pos_embed)

        # 在解码器的输入中，没有把位置嵌入添加到编码器的输出中
        node_dec_input, edge_dec_input = self.feature_reordering(remain_atom_pos_embed, remain_edge_pos_embed,
                                                                 mask_atom_pos_embed, mask_edge_pos_embed,
                                                                 node_mask_idx_list, node_remain_idx_list,
                                                                 edge_mask_idx_list, edge_remain_idx_list,
                                                                 node_embeddings, edge_embeddings)

        node_dec_output, edge_dec_output = self.decoders(node_dec_input, edge_dec_input, graph_batch)

        mask_node_pred, mask_edge_pred = self.get_mask_predict_features(node_dec_output, edge_dec_output,
                                                                        node_mask_idx_list, edge_mask_idx_list)

        # mask_node_target, mask_edge_target = self.get_mask_target_features(node_masked_batch, edge_masked_batch)

        # return mask_node_pred, mask_edge_pred, mask_node_target, mask_edge_target
        return mask_node_pred, mask_edge_pred


        # if self.embedding_output_type == 'atom':
        #     return {"atom_from_atom": output[0], "atom_from_bond": output[1],
        #             "bond_from_atom": None, "bond_from_bond": None}  # atom_from_atom, atom_from_bond
        # elif self.embedding_output_type == 'bond':
        #     return {"atom_from_atom": None, "atom_from_bond": None,
        #             "bond_from_atom": output[0], "bond_from_bond": output[1]}  # bond_from_atom, bond_from_bond
        # elif self.embedding_output_type == "both":
        #     return {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
        #             "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}

class BatmanNet_finetune(nn.Module):

    def __init__(self, args: Namespace):
        """
        Initialize MYGMAE_model class.
        :param args:
        """
        super(BatmanNet_finetune, self).__init__()

        # self.mask_ratio = args.mask_ratio
        self.hidden_size = args.hidden_size

        # self.embedding_output_type = args.embedding_output_type
        edge_dim = get_bond_fdim() + get_atom_fdim()
        node_dim = get_atom_fdim()

        # self.pos_embed = nn.Parameter(torch.zeros(1, num_atoms + 1, embed_dim), requires_grad=False)
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        self.encoders = GTransEncoder(args,
                                      hidden_size=args.hidden_size,
                                      edge_fdim=edge_dim,
                                      node_fdim=node_dim,
                                      dropout=args.dropout,
                                      activation=args.activation,
                                      num_mt_block=args.num_enc_mt_block,
                                      num_attn_head=args.num_attn_head,
                                      # atom_emb_output=self.embedding_output_type,
                                      bias=args.bias,
                                      cuda=args.cuda,
                                      res_connection=args.res_connection)


    def forward(self, graph_batch):
        """
        The forward function takes graph_batch as input and output a dict. The content of the dict is decided by
        self.embedding_output_type.

        :param graph_batch: the input graph batch generated by MolCollator.
        :return: a dict containing the embedding results.
        """

        # node_embeddings, edge_embeddings = self.encoders(graph_batch)
        # return node_embeddings, edge_embeddings
        output = self.encoders(graph_batch)
        return {"atom_from_atom": output[0][0], "bond_from_atom": output[0][1],
                "atom_from_bond": output[1][0], "bond_from_bond": output[1][1]}


class MyGMAE_Task(nn.Module):
    """
    The pretrain module.
    """
    def __init__(self, args, myGMAE_model):
        super(MyGMAE_Task, self).__init__()

        self.mygame = myGMAE_model


    @staticmethod
    def get_loss_func(args: Namespace) -> Callable:
        """
        The loss function generator.
        :param args: the arguments.
        :return: the loss fucntion for GroverTask.
        """
        def loss_func(mask_node_pred, mask_edge_pred, mask_node_target, mask_edge_target):
            """
            The loss function for GroverTask.
            :param preds: the predictions.
            :param targets: the targets.
            :param dist_coff: the default disagreement coefficient for the distances between different branches.
            :return:
            """
            node_atomtype_lossfc = nn.CrossEntropyLoss(reduction="mean")
            node_degree_lossfc = nn.CrossEntropyLoss(reduction="mean")
            node_charge_lossfc = nn.CrossEntropyLoss(reduction="mean")
            node_chirality_lossfc = nn.CrossEntropyLoss(reduction="mean")
            node_is_aromatic_lossfc = nn.BCEWithLogitsLoss(reduction="mean")

            edge_node_atom_type_lossfc = nn.CrossEntropyLoss(reduction="mean")
            edge_node_degree_lossfc = nn.CrossEntropyLoss(reduction="mean")
            edge_node_charge_lossfc = nn.CrossEntropyLoss(reduction="mean")
            edge_node_chirality_lossfc = nn.CrossEntropyLoss(reduction="mean")
            edge_node_is_aromatic_lossfc = nn.BCEWithLogitsLoss(reduction="mean")
            edge_bond_type_lossfc = nn.CrossEntropyLoss(reduction="mean")
            edge_stereo_confi_lossfc = nn.CrossEntropyLoss(reduction="mean")


            mask_node_target = [x.cuda() for x in mask_node_target]
            mask_edge_target = [x.cuda() for x in mask_edge_target]
            node_atomtype_loss = node_atomtype_lossfc(mask_node_pred[0], mask_node_target[0])
            node_degree_loss = node_degree_lossfc(mask_node_pred[1], mask_node_target[1])
            node_charge_loss = node_charge_lossfc(mask_node_pred[2], mask_node_target[2])
            node_chirality_loss = node_chirality_lossfc(mask_node_pred[3], mask_node_target[3])
            node_is_aromatic_loss = node_is_aromatic_lossfc(mask_node_pred[4].squeeze(dim=-1), mask_node_target[4])

            edge_node_atom_type_loss = edge_node_atom_type_lossfc(mask_edge_pred[0], mask_edge_target[0])
            edge_node_degree_loss = edge_node_degree_lossfc(mask_edge_pred[1], mask_edge_target[1])
            edge_node_charge_loss = edge_node_charge_lossfc(mask_edge_pred[2], mask_edge_target[2])
            edge_node_chirality_loss = edge_node_chirality_lossfc(mask_edge_pred[3], mask_edge_target[3])
            edge_node_is_aromatic_loss = edge_node_is_aromatic_lossfc(mask_edge_pred[4].squeeze(dim=-1), mask_edge_target[4])
            edge_bond_type_loss = edge_bond_type_lossfc(mask_edge_pred[5], mask_edge_target[5])
            edge_stereo_confi_loss = edge_stereo_confi_lossfc(mask_edge_pred[6], mask_edge_target[6])

            node_loss = node_atomtype_loss + node_degree_loss + node_charge_loss + node_chirality_loss + \
                        node_is_aromatic_loss
            edge_loss = edge_node_atom_type_loss + edge_node_degree_loss + edge_node_charge_loss + \
                        edge_node_chirality_loss + edge_node_is_aromatic_loss + edge_bond_type_loss + \
                        edge_stereo_confi_loss


            overall_loss = node_loss + edge_loss

            return overall_loss, node_loss, edge_loss

        return loss_func

    def forward(self, graph_batch, node_masked_batch, edge_masked_batch):
        """
        The forward function.
        :param graph_batch:
        :return:
        """
        mask_node_pred, mask_edge_pred = self.mygame(graph_batch, node_masked_batch, edge_masked_batch)
        return mask_node_pred, mask_edge_pred



class myGMAE_FinetuneTask(nn.Module):
    """
    The finetune
    """
    def __init__(self, args):
        super(myGMAE_FinetuneTask, self).__init__()

        self.hidden_size = args.hidden_size
        self.iscuda = args.cuda

        self.mygame = BatmanNet_finetune(args)

        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=self.hidden_size,
                                   attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out)
        else:
            self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)

        self.mol_atom_from_atom_ffn = self.create_ffn(args)
        self.mol_atom_from_bond_ffn = self.create_ffn(args)
        #self.ffn = nn.ModuleList()
        #self.ffn.append(self.mol_atom_from_atom_ffn)
        #self.ffn.append(self.mol_atom_from_bond_ffn)

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                first_linear_dim += args.features_dim
            else:
                first_linear_dim = args.hidden_size + args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # TODO: ffn_hidden_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    @staticmethod
    def get_loss_func(args):
        def loss_func(preds, targets,
                      dt=args.dataset_type,
                      dist_coff=args.dist_coff):

            if dt == 'classification':
                pred_loss = nn.BCEWithLogitsLoss(reduction='none')
            elif dt == 'regression':
                pred_loss = nn.MSELoss(reduction='none')
            else:
                raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

            # print(type(preds))
            # TODO: Here, should we need to involve the model status? Using len(preds) is just a hack.
            if type(preds) is not tuple:
                # in eval mode.
                return pred_loss(preds, targets)

            # in train mode.
            dist_loss = nn.MSELoss(reduction='none')
            # dist_loss = nn.CosineSimilarity(dim=0)
            # print(pred_loss)

            dist = dist_loss(preds[0], preds[1])
            pred_loss1 = pred_loss(preds[0], targets)
            pred_loss2 = pred_loss(preds[1], targets)
            return pred_loss1 + pred_loss2 + dist_coff * dist

        return loss_func

    def forward(self, batch, features_batch, output_wise):
        _, _, _, _, _, a_scope, _, _ = batch

        output = self.mygame(batch)


        if features_batch[0] is not None:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
            if self.iscuda:
                features_batch = features_batch.cuda()
            features_batch = features_batch.to(output["atom_from_atom"])
            if len(features_batch.shape) == 1:
                features_batch = features_batch.view([1, features_batch.shape[0]])
        else:
            features_batch = None



        mol_atom_from_bond_output = self.readout(output["atom_from_bond"], a_scope)
        mol_atom_from_atom_output = self.readout(output["atom_from_atom"], a_scope)

        if features_batch is not None:
            mol_atom_from_atom_output = torch.cat([mol_atom_from_atom_output, features_batch], 1)
            mol_atom_from_bond_output = torch.cat([mol_atom_from_bond_output, features_batch], 1)

        if self.training:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            return atom_ffn_output, bond_ffn_output
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)
                bond_ffn_output = self.sigmoid(bond_ffn_output)
            output = (atom_ffn_output + bond_ffn_output) / 2

        return output




class myGMAE_DDI_Task(nn.Module):
    """
    The DDI task
    """
    def __init__(self, args):
        super(myGMAE_DDI_Task, self).__init__()

        self.hidden_size = args.hidden_size
        self.iscuda = args.cuda

        self.mygame = BatmanNet_finetune(args)

        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=self.hidden_size,
                                   attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out)
        else:
            self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)

        self.mol_atom_from_atom_ffn = self.create_ffn(args)
        self.mol_atom_from_bond_ffn = self.create_ffn(args)

        self.classification = args.dataset_type == 'classification'
        if self.classification:
            self.sigmoid = nn.Sigmoid()

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        # Note: args.features_dim is set according the real loaded features data
        if args.features_only:
            first_linear_dim = args.features_size + args.features_dim
        else:
            if args.self_attention:
                first_linear_dim = args.hidden_size * args.attn_out
                # TODO: Ad-hoc!
                # if args.use_input_features:
                first_linear_dim += args.features_dim
            else:
                first_linear_dim = (args.hidden_size + args.features_dim)*2

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        # TODO: ffn_hidden_size
        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    @staticmethod
    def get_loss_func(args):
        def loss_func(preds, targets,
                      dt=args.dataset_type,
                      dist_coff=args.dist_coff):

            if dt == 'classification':
                pred_loss = nn.BCEWithLogitsLoss(reduction='none')
            elif dt == 'regression':
                pred_loss = nn.MSELoss(reduction='none')
            else:
                raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

            # print(type(preds))
            # TODO: Here, should we need to involve the model status? Using len(preds) is just a hack.
            if type(preds) is not tuple:
                # in eval mode.
                return pred_loss(preds, targets)

            # in train mode.
            dist_loss = nn.MSELoss(reduction='none')
            # dist_loss = nn.CosineSimilarity(dim=0)
            # print(pred_loss)

            dist = dist_loss(preds[0], preds[1])

            pred_loss1 = pred_loss(preds[0], targets)
            pred_loss2 = pred_loss(preds[1], targets)

            return pred_loss1 + pred_loss2 + dist_coff * dist

        return loss_func

    def forward(self, batch1, batch2, features1_batch, features2_batch):
        _, _, _, _, _, a_scope1, _, _ = batch1
        _, _, _, _, _, a_scope2, _, _ = batch2

        output1 = self.mygame(batch1)
        output2 = self.mygame(batch2)

        if features1_batch[0] is not None and features2_batch[0] is not None:
            features1_batch = torch.from_numpy(np.stack(features1_batch)).float()
            features2_batch = torch.from_numpy(np.stack(features2_batch)).float()
            if self.iscuda:
                features1_batch = features1_batch.cuda()
                features2_batch = features2_batch.cuda()
            features1_batch = features1_batch.to(output1["atom_from_atom"])
            features2_batch = features2_batch.to(output2["atom_from_atom"])
            if len(features1_batch.shape) == 1:
                features1_batch = features1_batch.view([1, features1_batch.shape[0]])
            if len(features2_batch.shape) == 1:
                features2_batch = features2_batch.view([1, features2_batch.shape[0]])
        else:
            features1_batch = None
            features2_batch = None



        mol_atom_from_bond_output1 = self.readout(output1["atom_from_bond"], a_scope1)
        mol_atom_from_atom_output1 = self.readout(output1["atom_from_atom"], a_scope1)

        mol_atom_from_bond_output2 = self.readout(output2["atom_from_bond"], a_scope2)
        mol_atom_from_atom_output2 = self.readout(output2["atom_from_atom"], a_scope2)

        if features1_batch is not None:
            mol_atom_from_atom_output1 = torch.cat([mol_atom_from_atom_output1, features1_batch], 1)
            mol_atom_from_bond_output1 = torch.cat([mol_atom_from_bond_output1, features1_batch], 1)
        if features2_batch is not None:
            mol_atom_from_atom_output2 = torch.cat([mol_atom_from_atom_output2, features2_batch], 1)
            mol_atom_from_bond_output2 = torch.cat([mol_atom_from_bond_output2, features2_batch], 1)

        atom_output = torch.cat([mol_atom_from_atom_output1, mol_atom_from_atom_output2], 1)
        bond_output = torch.cat([mol_atom_from_bond_output1, mol_atom_from_bond_output2], 1)

        if self.training:
            atom_ffn_output = self.mol_atom_from_atom_ffn(atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(bond_output)
            return atom_ffn_output, bond_ffn_output
        else:
            atom_ffn_output = self.mol_atom_from_atom_ffn(atom_output)
            bond_ffn_output = self.mol_atom_from_bond_ffn(bond_output)
            if self.classification:
                atom_ffn_output = self.sigmoid(atom_ffn_output)
                bond_ffn_output = self.sigmoid(bond_ffn_output)

            output = (atom_ffn_output + bond_ffn_output) / 2

        return output


