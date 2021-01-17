import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadedAttentionLayer(nn.Module):

    def __init__(self, dim_model, dim_k, dim_v, h):
        super(MultiHeadedAttentionLayer, self).__init__()

        self.dim_model = dim_model
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.h = h

        self.Q_linear = nn.Linear(dim_model, dim_k * h)
        self.K_linear = nn.Linear(dim_model, dim_k * h)
        self.V_linear = nn.Linear(dim_model, dim_v * h)

        self.out_linear = nn.Linear(self.h * dim_v, dim_model)

    def forward(self, Q, K, V, mask=None):

        b, len_q, len_k, len_v = Q.size(0), Q.size(1), K.size(1), V.size(1)

        Q_ = self.Q_linear(Q).view(b, len_q, self.h, self.dim_k).transpose(1, 2)
        K_ = self.K_linear(K).view(b, len_k, self.h, self.dim_k).transpose(1, 2)
        V_ = self.V_linear(V).view(b, len_v, self.h, self.dim_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        out = self.__attention(Q_, K_, V_, mask)
        out = out.transpose(1, 2).contiguous().view(b, len_q, -1)
        out = self.out_linear(out)

        return out

    @staticmethod
    def __attention(Q, K, V, mask=None):
        d_k = K.shape[0]
        att = (Q / np.sqrt(d_k)).matmul(K.transpose(-1, -2))
        if mask is not None:
            att = att.masked_fill(mask == 0, -float('inf'))
        att = F.softmax(att, dim=-1)
        out = att.matmul(V)

        return out


class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, dim_in, dim_ff, prob_dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_ff)
        self.fc2 = nn.Linear(dim_ff, dim_in)
        self.dropout = nn.Dropout(prob_dropout)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        #out = self.dropout(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    pw_ffn = PositionWiseFeedForwardLayer(2, 4)
    test_input = torch.randn(2, 10, 2)
    out = pw_ffn(test_input)

