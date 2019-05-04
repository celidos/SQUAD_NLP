import torch
import torch.nn as nn
import torch.nn.functional as F

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
        batch_size, c_len, _ = c.size()

        c = F.dropout(c, self.drop_prob, self.training) # (bs, c_len, hid_size)
        w_c = self.c_weight(c).unsqueeze(1) # (bs, 1, c_len, out_size)

        w_word = self.word_weight(c).unsqueeze(2)# (bs, c_len, 1, out_size)
        tanh = self.tanh(w_c + w_word)  # (bs, c_len, c_len, out_size)

        a = self.v_weight(tanh)  # (bs, c_len, c_len, 1)
        a = F.softmax(a, dim=2)  # (bs, c_len, c_len, 1)

        a = a.squeeze(3)   # (bs, c_len, c_len)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        h = torch.bmm(a, c)   # (bs, c_len, hid_size)

        return h
