import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util import masked_softmax
import layers

class DoubleCrossAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(DoubleCrossAttention, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        
        alpha = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len) = 
        beta = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)



        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        A = torch.bmm(alpha, q)
        # (bs, q_len, c_len) x (bs, c_len, hid_size) => (bs, q_len, hid_size)
        B = torch.bmm(beta.transpose(1, 2), c)

        # second level cross attention matrix N × M
        # (bs, c_len, hid_size) x (bs, q_len, hid_size)^T -> (bs, c_len, q_len)
        R = torch.bmm(A, torch.transpose(B, 1, 2)) # (bs, c_len, q_len)
        
        gamma = F.softmax(R, dim=2) # (bs, c_len, q_len)
        
        #print('gamma shape = ', gamma.shape)
        #print('B shape = ', B.shape)
        
        
        # (bs, q_len, c_len) x ()
        D = torch.bmm(gamma, B) # (bs, q_len, hid_size)

        x = torch.cat([c, A, D], dim=2)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        s = torch.bmm(c, torch.transpose(q, 1, 2)) # (bs, c_len, q_len)

        return s

class BiDAFOutput_att(nn.Module):
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
    def __init__(self, hidden_size, att_put_h_size, drop_prob):
        super(BiDAFOutput_att, self).__init__()
        self.att_linear_1 = nn.Linear(att_put_h_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = layers.RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(att_put_h_size, 1)
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
