import layers
import layers_selfAtt
import torch
import torch.nn as nn


class BiDAF_selfAtt(nn.Module):
    """Baseline BiDAF model with Self-attention
    Based on https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf

    Run dev:
    python train_dev.py -n selfAtt --hidden_size 50

    Run:
    python train.py -n selfAtt --hidden_size 20

    (about 4Gb of GPU mem)
    """

    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF_selfAtt, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.selfAtt = layers_selfAtt.SelfAttention(in_size=hidden_size, hidden_size=8*hidden_size,
                                                    drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs #context
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs #query
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)  # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)      # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)      # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)      # (batch_size, c_len, 8 * hidden_size)

        att = self.selfAtt(att, c_mask)     # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)  # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
