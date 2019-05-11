import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import layers

class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """

    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = layers.RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(4 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2


class SelfAttention(nn.Module):
    """Self-attention

    """
    def __init__(self, in_size, hidden_size, drop_prob=0.1):
        super(SelfAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Linear(hidden_size, in_size, bias=False)
        self.word_weight = nn.Linear(hidden_size, in_size, bias=False)
        self.v_weight = nn.Linear(in_size, 1, bias=False)

        self.tanh = nn.Tanh()
        for weight in (self.c_weight, self.word_weight, self.v_weight):
            nn.init.xavier_uniform_(weight.weight)

    def forward(self, c, c_mask):
        batch_size, c_len, hidden_size = c.size()

        g_tanh = (self.c_weight(c)).tanh()
        gt = self.v_weight.forward(g_tanh).squeeze(2)

        gt_prop = masked_softmax(gt, c_mask, dim = 1)
        gt_prop = gt_prop.unsqueeze(2)
        c_gt = c*gt_prop
        return c_gt

        # h = []
        #
        # for i in range(c_len):
        #     word = c[:, i, :]
        #     w_c = self.c_weight(c)
        #     w_word = self.word_weight(word).unsqueeze(1)
        #     tanh = (w_c + w_word).tanh()
        #     a = self.v_weight(tanh).squeeze(2)
        #     a = masked_softmax(a, c_mask, dim=1).unsqueeze(1)
        #     new_h = torch.bmm(a, c).squeeze(1)
        #     h.append(new_h)
        #
        # h = torch.stack(h, dim = 0)

        # batch_size, c_len, _ = c.size()
        #
        # c = F.dropout(c, self.drop_prob, self.training) # (bs, c_len, hid_size)
        # w_c = self.c_weight(c).unsqueeze(1) # (bs, 1, c_len, out_size)
        #
        # w_word = self.word_weight(c).unsqueeze(2)# (bs, c_len, 1, out_size)
        # tanh = self.tanh(w_c + w_word)  # (bs, c_len, c_len, out_size)
        #
        # a = self.v_weight(tanh)  # (bs, c_len, c_len, 1)
        # a = F.softmax(a, dim=2)  # (bs, c_len, c_len, 1)
        #
        # a = a.squeeze(3)   # (bs, c_len, c_len)
        # # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        # h = torch.bmm(a, c)   # (bs, c_len, hid_size)
        return h



