"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)


class Embedding(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


# custom
class EmbeddingWithCharLevel(nn.Module):
    """
    
    """
    def __init__(self, word_vectors, hidden_size, drop_prob, \
                 char_dict_size, char_emb_size, \
                 conv_kernel_size, conv_depth1, \
                 conv_output_hidden_size):
        super(EmbeddingWithCharLevel, self).__init__()
        self.drop_prob = drop_prob
        self.conv_output_hidden_size = conv_output_hidden_size
        
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding(char_dict_size, char_emb_size, max_norm=1.0, norm_type=2.0).cuda()
        
        self.conv1 =  nn.Conv2d(in_channels = 1, 
                                out_channels = conv_depth1, 
                                kernel_size = (conv_kernel_size, char_emb_size),
                                padding=[1, 0])
        self.conv2 =  nn.Conv2d(in_channels = 1, 
                                out_channels = conv_output_hidden_size, 
                                kernel_size = (conv_kernel_size, conv_depth1),
                                padding=[1, 0])
        
        self.wproj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.cproj = nn.Linear(conv_output_hidden_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, 2*hidden_size) # *2

    def forward(self, x_words, x_chars):
        emb_words = self.word_embed(x_words)   # (batch_size, seq_len, word_embed_size)
        
#         print('x_char shape = ', x_chars.shape)
        emb_char = self.char_embed(x_chars)    
        
#         print('emb_char shape = ', emb_char.shape)
        
        d1, d2, d3 = emb_char.shape[0], emb_char.shape[1], emb_char.shape[2]
        
        emb_char_reshaped = torch.reshape(emb_char, (d1 * d2, emb_char.shape[2], emb_char.shape[3]))
        emb_char_reshaped = emb_char_reshaped.unsqueeze(1)  # (batch_size * seq_len, 1, max_word_len, char_emb_size)
        
#         print('emb_char_reshaped.shape = ', emb_char_reshaped.shape)
        
        cnv = self.conv1(emb_char_reshaped)
        cnv = cnv.permute((0, 3, 2, 1))
#         print('cnv shape = ', cnv.shape)
        cnv = self.conv2(cnv)
#         print('cnv shape = ', cnv.shape)
        cemb = torch.reshape(cnv, (d1, d2, self.conv_output_hidden_size, d3))
#         print('cnv shape = ', cemb.shape)
        
#         #max pool
        cemb = cemb.max(dim=3)[0]
        
        wemb = F.dropout(emb_words, self.drop_prob, self.training)
        
        wemb = self.wproj(wemb)  # (batch_size, seq_len, hidden_size)
#         print('cemb shape = ', cemb.shape)
        cemb = self.cproj(cemb)  # (batch_size, seq_len, hidden_size)
        
        emb = torch.cat((wemb, cemb), 2)
        
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb
    
        
        
class EmbeddingWithCharLevel2_franky(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob, \
                 char_dict_size, char_emb_size, \
                 conv_kernel_size, conv_depth1, \
                 conv_output_hidden_size):
        super(EmbeddingWithCharLevel2_franky, self).__init__()
        
        self.drop_prob = drop_prob
        self.conv_output_hidden_size = conv_output_hidden_size
        
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding(char_dict_size, char_emb_size).cuda()
        
        self.wemb_dim = word_vectors.size(1)
        
        self.conv2d = nn.Conv2d(char_emb_size, hidden_size, kernel_size = (1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(self.wemb_dim + hidden_size, hidden_size, bias=False)
        self.high = HighwayEncoder(2, hidden_size)
        self.dropout_w = drop_prob
        self.dropout_c = drop_prob

    def forward(self, x_words, x_chars):
        
        
        emb_words = self.word_embed(x_words)   # (batch_size, seq_len, word_embed_size)
        
#         print('x_char shape = ', x_chars.shape)
        emb_char = self.char_embed(x_chars)    
        
        ch_emb = emb_char.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)

        wd_emb = F.dropout(emb_words, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        
        #print('embshape=', emb.shape)
        emb = emb.transpose(1, 2)
        emb = self.high(emb)
        return emb  

   
class EmbeddingWithCharLevel2(nn.Module):
    def __init__(self, word_vectors, hidden_size, drop_prob, \
                 char_dict_size, char_emb_size, \
                 conv_kernel_size, conv_depth1, \
                 conv_output_hidden_size):
        super(EmbeddingWithCharLevel2, self).__init__()
        
        self.drop_prob = drop_prob
        self.conv_output_hidden_size = conv_output_hidden_size
        
        self.word_embed = nn.Embedding.from_pretrained(word_vectors)
        self.char_embed = nn.Embedding(char_dict_size, char_emb_size).cuda()
        
        self.wemb_dim = word_vectors.size(1)
        
        self.conv2d = nn.Conv2d(char_emb_size, hidden_size, kernel_size = (1,5), padding=0, bias=True)
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(self.wemb_dim + hidden_size, 2*hidden_size, bias=False)
        self.high = HighwayEncoder(2, 2*hidden_size)
        self.dropout_w = drop_prob
        self.dropout_c = drop_prob

    def forward(self, x_words, x_chars):
        
        
        emb_words = self.word_embed(x_words)   # (batch_size, seq_len, word_embed_size)
        
#         print('x_char shape = ', x_chars.shape)
        emb_char = self.char_embed(x_chars)    
        
        ch_emb = emb_char.permute(0, 3, 1, 2)
        ch_emb = F.dropout(ch_emb, p=self.dropout_c, training=self.training)
        ch_emb = self.conv2d(ch_emb)
        ch_emb = F.relu(ch_emb)
        ch_emb, _ = torch.max(ch_emb, dim=3)

        wd_emb = F.dropout(emb_words, p=self.dropout_w, training=self.training)
        wd_emb = wd_emb.transpose(1, 2)
        emb = torch.cat([ch_emb, wd_emb], dim=1)
        emb = self.conv1d(emb)
        
        #print('embshape=', emb.shape)
        emb = emb.transpose(1, 2)
        emb = self.high(emb)
        return emb   
    
class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

class HighwayEncoder_franky(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, input_size, hidden_size):
        super(HighwayEncoder_franky, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size) if i > 0 else nn.Linear(input_size, hidden_size)
                                         for i in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size) if i > 0 else nn.Linear(input_size, hidden_size)
                                    for i in range(num_layers)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x

class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths, batch_first=True)

        # Apply RNN
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
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
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

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

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


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
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
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
