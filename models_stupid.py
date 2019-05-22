import layers
import layers_selfAtt
import layers_att
import torch
import torch.nn as nn
from layers_att import DoubleCrossAttention, BiDAFOutput_att


class BiDAF_monster(nn.Module):
    """Baseline BiDAF model with Self-attention
    Based on https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

    Run dev:
    python train_dev.py -n selfAtt --hidden_size 50

    Run:
    python train.py -n selfAtt --hidden_size 20

    (about 4Gb of GPU mem)
    """

    def __init__(self, word_vectors, hidden_size,
                char_dict_size, char_emb_size, \
                 conv_kernel_size, conv_depth1, \
                 conv_output_hidden_size, drop_prob=0.):
        super(BiDAF_monster, self).__init__()
        self.emb = layers.EmbeddingWithCharLevel2(word_vectors=word_vectors,
                                                 hidden_size=hidden_size,
                                                 drop_prob=drop_prob,
                                                 char_dict_size=char_dict_size,
                                                 char_emb_size=char_emb_size,
                                                 conv_kernel_size=conv_kernel_size, 
                                                 conv_depth1=conv_depth1,
                                                 conv_output_hidden_size=conv_output_hidden_size)

        self.enc = layers.RNNEncoder(input_size=2*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        #self.att = DoubleCrossAttention(hidden_size=hidden_size,
        #                                drop_prob=drop_prob)
        self.att = layers.BiDAFAttention(hidden_size=2* hidden_size,
                                         drop_prob=drop_prob)

        #self.selfAtt = layers_selfAtt.SelfAttention(in_size=6*hidden_size, hidden_size=hidden_size,
        #                                            drop_prob=drop_prob)
        self.selfAtt = layers_selfAtt.SelfAttention2(n_head=8, d_model=8*hidden_size, d_k=64, d_v=64, dropout=0.1)

        self.mod = layers.RNNEncoder(input_size=8*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)
        self.out = BiDAFOutput_att(hidden_size=hidden_size,
                                   att_put_h_size=8*hidden_size,
                                   drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs #context
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs #query
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)      # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)      # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)      # (batch_size, c_len, 8 * hidden_size)

        

        # att
        #att = self.selfAtt(att)     # (batch_size, c_len, 8 * hidden_size)
        att = self.selfAtt(att, att, att)     # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
