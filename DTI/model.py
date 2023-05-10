from graph_bert import *
from BatmanNet.model.models import BatmanNet_finetune
from BatmanNet.model.layers import Readout, GTransEncoder, GTransDecoder
from BatmanNet.util.nn_utils import get_activation_function
from argparse import Namespace
import numpy as np
class CompoundProteinInteractionPrediction(nn.Module):
    def __init__(self, args, n_word, dim=768, window=11, layer_cnn=3, layer_output=3, graph_pooling='mean'):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.mygame = BatmanNet_finetune(args)
        if args.self_attention:
            self.readout = Readout(rtype="self_attention", hidden_size=self.hidden_size,
                                   attn_hidden=args.attn_hidden,
                                   attn_out=args.attn_out)
        else:
            self.readout = Readout(rtype="mean", hidden_size=self.hidden_size)
        self.mol_atom_from_atom_ffn = self.create_ffn(args, dim)
        self.mol_atom_from_bond_ffn = self.create_ffn(args, dim)



        self.embed_word = nn.Embedding(n_word, dim)

        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        self.W_interaction = nn.Linear(2*dim, 1)
        self.layer_cnn = layer_cnn
        self.layer_output = layer_output
        self.dummy=False
        self.sigmoid = nn.Sigmoid()

    def create_ffn(self, args: Namespace, dim):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        # Note: args.features_dim is set according the real loaded features data

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
                nn.Linear(first_linear_dim, dim)
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
                nn.Linear(args.ffn_hidden_size, dim),
            ])

        # Create FFN model
        return nn.Sequential(*ffn)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""
        # x: compound, xs: protein (n,len,hid)

        xs = torch.unsqueeze(xs, 1) # (n,1,len,hid)
        # print('xs',xs.shape)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        # print('xs1',xs.shape) #(n,1,len,hid)
        xs = torch.squeeze(xs, 1)
        # print('xs2',xs.shape)# (n,len,hid)

        h = torch.relu(self.W_attention(x)) #n,hid
        hs = torch.relu(self.W_attention(xs))#n,len,hid
        weights = torch.tanh(torch.bmm(h.unsqueeze(1),hs.permute(0,2,1))) #torch.tanh(F.linear(h, hs))#n,len
        ys = weights.permute(0,2,1) * hs #n,l,h
        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.mean(ys, 1)

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

    def forward(self, inputs):

        smiles_batch, batch, features_batch, mask, targets, protein_batch, protein_len_batch = inputs
        _, _, _, _, _, a_scope, _, _ = batch

        """Compound vector with BatmanNer encoder."""
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


        atom_ffn_output = self.mol_atom_from_atom_ffn(mol_atom_from_atom_output)
        bond_ffn_output = self.mol_atom_from_bond_ffn(mol_atom_from_bond_output)

        """Protein vector with attention-CNN."""
        protein_batch = torch.tensor(protein_batch, dtype=torch.int64)
        protein_batch = protein_batch.cuda()
        word_vectors = self.embed_word(protein_batch)
        # print('word',word_vectors.shape) #(len,hid)

        protein_vector1 = self.attention_cnn(atom_ffn_output,
                                            word_vectors, self.layer_cnn)
        # print('protein',[protein_vector.shape]) #(1,hid)
        """Concatenate the above two vectors and output the interaction."""
        cat_vector1 = torch.cat((atom_ffn_output, protein_vector1), 1)
        for j in range(self.layer_output):
            cat_vector1 = torch.relu(self.W_out[j](cat_vector1))
        interaction1 = self.W_interaction(cat_vector1)

        protein_vector2 = self.attention_cnn(bond_ffn_output,
                                            word_vectors, self.layer_cnn)
        """Concatenate the above two vectors and output the interaction."""
        cat_vector2 = torch.cat((bond_ffn_output, protein_vector2), 1)
        for j in range(self.layer_output):
            cat_vector2 = torch.relu(self.W_out[j](cat_vector2))
        interaction2 = self.W_interaction(cat_vector2)

        if self.training:
            return interaction1, interaction2
        else:
            interaction1 = self.sigmoid(interaction1)
            interaction2 = self.sigmoid(interaction2)
            output = (interaction1 + interaction2) / 2
            return output




    def from_pretrain(self, model_file):
        self.mygame.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))


