import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SelfAttention(nn.Module):
    """Self-attention

    """
    def __init__(self, in_size, hidden_size, drop_prob=0.1):
        super(SelfAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Linear(in_size, hidden_size, bias=False).cuda()   # 6*h x h 
        self.word_weight = nn.Linear(in_size, hidden_size, bias=False).cuda() # 6*h x h
        self.v_weight = nn.Linear(hidden_size, 1, bias=False).cuda()              # h x 1

        #self.tanh = nn.Tanh()
        for weight in (self.c_weight, self.word_weight, self.v_weight):
            nn.init.xavier_uniform_(weight.weight)

    def forward(self, c):
        batch_size, c_len, _ = c.size()

        c = F.dropout(c, self.drop_prob, self.training) # (bs, c_len, hid_size)
        w_c = self.c_weight(c).unsqueeze(1).cuda() # (bs, 1, c_len, out_size)

        w_word = self.word_weight(c).unsqueeze(2).cuda()# (bs, c_len, 1, out_size)
        #print('w_word.shape = ', w_word.shape)
        #print('w_c.shape = ', w_c.shape)
        #print('w_c + w_word shape = ', (w_c + w_word).shape)
        tanh = (w_c + w_word).tanh().cuda()  # (bs, c_len, c_len, out_size)
        #tanh = self.tanh(w_c + w_word)  # (bs, c_len, c_len, out_size)

        #a = self.v_weight(tanh)  # (bs, c_len, c_len, 1)
        #a = F.softmax(a, dim=2)  # (bs, c_len, c_len, 1)

        #a = a.squeeze(3)   # (bs, c_len, c_len)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        #h = torch.bmm(a, c)   # (bs, c_len, hid_size)

        #del tanh    

        return None#h
        
        
# ----------------------------------------------------------------------
class SelfAttention2(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        
        # ---------------------
        
        self.temperature = np.power(d_k, 0.5)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
        # ----------------------
        
        self.dropout = nn.Dropout(dropout)


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        # -------------------------------------------
        
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        
        # ----------------------------------------------

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output
