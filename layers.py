import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

from sublayers import PositionWiseFeedForwardLayer, MultiHeadedAttentionLayer


class TokenEmbeddingLayer(nn.Module):

    def __init__(self, n_token, dim_model):
        super(TokenEmbeddingLayer, self).__init__()
        self.dim_model = dim_model
        self.embed = nn.Embedding(n_token, dim_model)

    def forward(self, x):
        out = self.embed(x)
        out = out * np.sqrt(self.dim_model)
        return out


class FeatureEmbeddingLayer(nn.Module):

    def __init__(self, dim_feature, dim_model):
        super(FeatureEmbeddingLayer, self).__init__()
        self.dim_model = dim_model
        self.embed = nn.Linear(dim_feature, dim_model)

    def forward(self, x):
        out = self.embed(x)
        out = out * np.sqrt(self.dim_model)
        return out


class SegmentEmbeddingLayer(nn.Module):

    def __init__(self, n_typetoken, n_segpostoken, dim_model):
        super(SegmentEmbeddingLayer, self).__init__()
        self.dim_model = dim_model
        self.token_embed = nn.Embedding(n_typetoken, dim_model)
        self.segpos_embed = nn.Embedding(n_segpostoken, dim_model)

    #typeid: 0 for s , 1 for c
    #segmentid: 0 for s, 1 for p,2 for n
    #def forward(self, x, segmentid):
    def forward(self, x, typeid, segmentid):
        l = x.shape[1]
        for i in range(l):
            x[:, i, :] += self.token_embed(torch.tensor(typeid).cuda(x.device))
            x[:, i, :] += self.segpos_embed(torch.tensor(segmentid).cuda(x.device))
            #True

        return x


class PositionalEncodingLayer(nn.Module):

    def __init__(self, dim_model, prob_dropout=0.1, len_seq=250):
        super(PositionalEncodingLayer, self).__init__()
        self.dim_model = dim_model
        self.len_seq = len_seq
        self.dropout = nn.Dropout(prob_dropout)
        self.register_buffer('pos_mat', self.__get_sinusoid_mat())
        self.pos_embed = nn.Embedding(len_seq, dim_model)
        #self.type_embed = nn.Parameter(torch.rand(dim_model))
        self.type_embed = nn.Embedding(2, dim_model)

    def __get_sinusoid_mat(self) -> torch.tensor:

        def __get_angle_vec(pos):
            return pos / np.power(10000, [2 * (pos_j // 2) / self.dim_model for pos_j in range(self.dim_model)])

        pos_mat = np.array([__get_angle_vec(pos) for pos in range(self.len_seq)])
        pos_mat[:, 0::2] = np.sin(pos_mat[:, 0::2])
        pos_mat[:, 1::2] = np.cos(pos_mat[:, 1::2])

        return torch.from_numpy(pos_mat).unsqueeze(0)

    #istype=-1 no type embedding; 0 source embedding; 1 target embedding
    def forward(self, x, istype=-1):
        out = x + self.pos_mat[:, :x.size(1)].clone().detach().type_as(x)
        
        if istype != -1:
            l = out.shape[1]
            for i in range(l):
                out[:, i, :] += self.type_embed(torch.tensor(istype).cuda(out.device))
            
        return self.dropout(out)

        #l = x.shape[1]
        #for i in range(l):
        #    x[:, i, :] += self.embed(torch.tensor(i).cuda(x.device))
        #return self.dropout(x)


class ResidualConnectionLayer(nn.Module):

    def __init__(self, dim_model, prob_dropout=0.1, add_sublayer=True):
        super(ResidualConnectionLayer, self).__init__()
        self.add_sublayer = add_sublayer
        self.norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(prob_dropout)

    def forward(self, x, sublayer):
        out = self.norm(x)
        out = sublayer(out)
        out = self.dropout(out)
        if self.add_sublayer:
            return x + out
        else:
            return out


class BaseLayer(nn.Module):

    def __init__(self, dim_model, dim_k, dim_v, h, dim_ff, prob_dropout):
        super(BaseLayer, self).__init__()
        self._dim_model = dim_model
        self._dim_k = dim_k
        self._dim_v = dim_v
        self._h = h
        self._dim_ff = dim_ff
        self._prob_dropout = prob_dropout


class BasicAttentionLayer(BaseLayer):

    def __init__(self, dim_model, dim_k, dim_v, h, dim_ff, prob_dropout):
        super(BasicAttentionLayer, self).__init__(dim_model, dim_k, dim_v, h, dim_ff, prob_dropout)
        self.self_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h)
        self.rc1 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.ff = PositionWiseFeedForwardLayer(dim_model, dim_ff)
        self.rc2 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.norm = nn.LayerNorm(dim_model)


    def forward(self, x, mask=None):
        out = self.rc1(x, lambda item: self.self_att(item, item, item, mask))
        out = self.rc2(out, self.ff)
        out = self.norm(out)
        return out


class ForwardCrossAttentionLayer(BaseLayer):

    def __init__(self, dim_model, dim_k, dim_v, h, dim_ff, prob_dropout):
        super(ForwardCrossAttentionLayer, self).__init__(dim_model, dim_k, dim_v, h, dim_ff, prob_dropout)
        self.self_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h)
        self.rc1 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.cross_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h)
        self.rc2 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.ff = PositionWiseFeedForwardLayer(dim_model, dim_ff)
        self.rc3 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x, memory_x, mask=None, memory_mask=None):
        out = self.rc1(x, lambda item: self.self_att(item, item, item, mask))
        out = self.rc2(out, lambda item: self.cross_att(item, memory_x, memory_x, memory_mask))
        out = self.rc3(out, self.ff)
        out = self.norm(out)
        return out


class BackwardCrossAttentionLayer(BaseLayer):

    def __init__(self, dim_model, dim_k, dim_v, h, dim_ff, prob_dropout):
        super(BackwardCrossAttentionLayer, self).__init__(dim_model, dim_k, dim_v, h, dim_ff, prob_dropout)
        #self.self_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h)
        #self.rc1 = ResidualConnectionLayer(dim_model, prob_dropout)
        #self.ff1 = PositionWiseFeedForwardLayer(dim_model, dim_ff)        
        #self.rc2 = ResidualConnectionLayer(dim_model, prob_dropout)

        self.cross_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h)
        self.rc3 = ResidualConnectionLayer(dim_model, prob_dropout, False)
        self.ff2 = PositionWiseFeedForwardLayer(dim_model, dim_ff)
        self.rc4 = ResidualConnectionLayer(dim_model, prob_dropout, False)
        self.norm = nn.LayerNorm(dim_model)

    def forward(self, x, memory_x, mask=None, memory_mask=None):
        #out = self.rc1(x, lambda item: self.self_att(item, item, item, mask))
        #out = self.rc2(out, self.ff1)
        out = self.rc3(x, lambda item: self.cross_att(memory_x, item, item, memory_mask))
        out = self.rc4(out, self.ff2)
        out = self.norm(out)
        return out


class DoubleForwardCrossAttentionLayer(BaseLayer):

    def __init__(self, dim_model, dim_k, dim_v, h, dim_ff, prob_dropout):
        super(DoubleForwardCrossAttentionLayer, self).__init__(dim_model, dim_k, dim_v, h, dim_ff, prob_dropout)
        self.self_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h)
        self.rc1 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.context_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h)
        self.rc2 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.encoder_att = MultiHeadedAttentionLayer(dim_model, dim_k, dim_v, h)
        self.rc3 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.ff = PositionWiseFeedForwardLayer(dim_model, dim_ff)
        self.rc4 = ResidualConnectionLayer(dim_model, prob_dropout)

    def forward(self, x, context_x, encoder_x, mask=None, context_mask=None, encoder_mask=None):
        out = self.rc1(x, lambda item: self.self_att(item, item, item, mask))
        out = self.rc2(out, lambda item: self.context_att(item, context_x, context_x, context_mask))
        out = self.rc3(out, lambda item: self.encoder_att(item, encoder_x, encoder_x, encoder_mask))
        out = self.rc4(out, self.ff)
        return out

class GateGRUSelectionLayer(nn.Module):

    def __init__(self, dim_model, dim_ff, prob_dropout):
        super(GateGRUSelectionLayer, self).__init__()
        self.reset = nn.Linear(dim_model*2, dim_model)
        self.update = nn.Linear(dim_model*2, dim_model)
        self.proposal = nn.Linear(dim_model*2, dim_model)
        # self.gate = nn.Parameter(torch.rand(dim_model))

    def forward(self, x_1, x_2, *args):
        reset = torch.sigmoid(self.reset(torch.cat([x_1, x_2], -1)))
        update = torch.sigmoid(self.update(torch.cat([x_1, x_2], -1)))
        proposal = torch.tanh(self.proposal(torch.cat([reset * x_1, x_2], -1)))
        out = (1 - update) * x_1 + update * proposal
        return out


class GateContextSelectionLayer(nn.Module):

    def __init__(self, dim_model, dim_ff, prob_dropout):
        super(GateContextSelectionLayer, self).__init__()
        self.source = nn.Linear(dim_model, dim_model)
        self.context = nn.Linear(dim_model, dim_model)

    def forward(self, x_1, x_2, *args):
        update = torch.sigmoid(self.source(x_1) + self.context(x_2))
        out = (1 - update) * x_1 + update * x_2
        return out


class GateSelectionLayer(nn.Module):

    def __init__(self, dim_model, dim_ff, prob_dropout):
        super(GateSelectionLayer, self).__init__()
        self.gate = nn.Parameter(torch.rand(dim_model))
        self.ff = PositionWiseFeedForwardLayer(dim_model, dim_ff)
        self.rc3 = ResidualConnectionLayer(dim_model, prob_dropout)
        self.norm = nn.LayerNorm(dim_model)


    def forward(self, x_1, x_2):
        out = x_1 * self.gate + x_2 * (1 - self.gate)
        #out = self.rc3(out, self.ff)
        #out = self.norm(out)
        return out


class ConcatFusionLayer(nn.Module):

    def __init__(self, dim_model,
                 voc_size, dout_p):
        super(ConcatFusionLayer, self).__init__()
        self.linear = nn.Linear(dim_model, voc_size)
        self.dropout = nn.Dropout(dout_p)
        self.linear2 = nn.Linear(voc_size, voc_size)

    def forward(self, x):
        x = self.linear(x)
        x = self.linear2(self.dropout(F.relu(x)))
        return F.log_softmax(x, dim=-1)


class __Test:

    @staticmethod
    def positional_encoding_test():
        plt.figure(figsize=(15, 5))
        pe = PositionalEncodingLayer(20)
        out = pe(Variable(torch.zeros(1, 100, 20)))
        plt.plot(np.arange(100), out[0, :, 8:12].data.numpy())
        plt.legend([f"dim {p}" for p in [8, 9, 10, 11]])
        plt.show()


if __name__ == '__main__':
    __Test.positional_encoding_test()
    pass
