import layers
import torch
import torch.nn as nn
import layers_franky
from layers_att import DoubleCrossAttention, BiDAFOutput_att

class BiDAF_franky(nn.Module):
    """Baseline BiDAF model for SQuAD.

    """
    def __init__(self, word_vectors, hidden_size,
                char_dict_size, char_emb_size, \
                 conv_kernel_size, conv_depth1, \
                 conv_output_hidden_size, drop_prob=0.2):
        super(BiDAF_franky, self).__init__()
        
        
        word_vectors, hidden_size, drop_prob, \
                 char_dict_size, char_emb_size, \
                 conv_kernel_size, conv_depth1, \
                 conv_output_hidden_size
        self.emb = layers.EmbeddingWithCharLevel2_franky(word_vectors=word_vectors,
                                                 hidden_size=hidden_size,
                                                 drop_prob=drop_prob,
                                                 char_dict_size=char_dict_size,
                                                 char_emb_size=char_emb_size,
                                                 conv_kernel_size=conv_kernel_size, 
                                                 conv_depth1=conv_depth1,
                                                 conv_output_hidden_size=conv_output_hidden_size)

        self.enc2 = layers_franky.EncoderBlock(conv_num=2,
                                               d_model=hidden_size,
                                               num_head=4,
                                               k=5)
        self.enc1 = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.compr = layers.Initialized_Conv1d(in_channels=3*hidden_size, out_channels=hidden_size*2, kernel_size=1, relu=False, bias=False)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)


        self.self_att = layers_franky.SelfAttention(d_model=8*hidden_size, num_head=4, dropout=drop_prob)
                                         

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = BiDAFOutput_att(hidden_size=hidden_size,
                                   att_put_h_size=8*hidden_size,
                                   drop_prob=drop_prob)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        c_enc1 = self.enc1(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc1 = self.enc1(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)
        
        c_enc2 = self.enc2(c_emb.transpose(1, 2), c_mask)    # ?
        q_enc2 = self.enc2(q_emb.transpose(1, 2), q_mask)    # ?
        
        #print(c_enc1.shape)
        #print(c_enc2.shape)
        
        c_enc = torch.cat([c_enc1, c_enc2.transpose(1, 2)], dim=2)
        q_enc = torch.cat([q_enc1, q_enc2.transpose(1, 2)], dim=2)
        
        c_enc = self.compr(c_enc.transpose(1, 2)).transpose(1, 2)
        q_enc = self.compr(q_enc.transpose(1, 2)).transpose(1, 2)
        
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        #print(att.shape)

        self_att = self.self_att(att.transpose(1, 2), c_mask).transpose(1, 2)

        mod = self.mod(self_att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
