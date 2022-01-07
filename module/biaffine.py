import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
# from torch.nn import functional, init

def get_tensor_np(t):
    return t.data.cpu().numpy()

def orthonormal_initializer(output_size, input_size):
    """
    adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
    """
    print(output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in range(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
                    np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))

def timestep_dropout(inputs, p=0.5, batch_first=True):
    '''
    :param inputs: (bz, time_step, feature_size)
    :param p: probability p mask out output nodes
    :param batch_first: default True
    :return:
    '''
    if not batch_first:
        inputs = inputs.transpose(0, 1)

    batch_size, time_step, feature_size = inputs.size()
    drop_mask = inputs.data.new_full((batch_size, feature_size), 1-p)
    drop_mask = torch.bernoulli(drop_mask).div(1 - p)
    # drop_mask = drop_mask.unsqueeze(-1).expand((-1, -1, time_step)).transpose(1, 2)
    drop_mask = drop_mask.unsqueeze(1)
    return inputs * drop_mask

class NonlinearMLP(nn.Module):
    def __init__(self, in_feature, out_feature, activation=nn.ReLU(), bias=True):
        super(NonlinearMLP, self).__init__()

        if activation is None:
            self.activation = lambda x: x
        else:
            assert callable(activation)
            self.activation = activation

        self.bias = bias
        self.linear = nn.Linear(in_features=in_feature,
                                out_features=out_feature,
                                bias=bias)

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        if self.bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, inputs):
        linear_out = self.linear(inputs)
        return self.activation(linear_out)


class Biaffine(nn.Module):
    def __init__(self, in_features,
                 out_features=1,
                 bias=(True, True)  # True = 1  False = 0
                 ):
        super(Biaffine, self).__init__()
        self.in_features = in_features  # mlp_arc_size / mlp_label_size
        self.out_features = out_features  # 1 / rel_size
        self.bias = bias

        # arc / label: mlp_size + 1
        self.linear_input_size = in_features + bias[0]
        # arc: mlp_size
        # label: (mlp_size + 1) * rel_size
        self.linear_output_size = out_features * (in_features + bias[1])

        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.zeros_(self.linear.weight)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        if self.bias[0]:
            ones = input1.data.new_ones(batch_size, len1, 1)
            input1 = torch.cat((input1, ones), dim=-1)
            # dim1 += 1
        if self.bias[1]:
            ones = input2.data.new_ones(batch_size, len2, 1)
            input2 = torch.cat((input2, ones), dim=-1)
            # dim2 += 1

        # (bz, len1, dim1+1) -> (bz, len1, linear_output_size)
        affine = self.linear(input1)

        # (bz, len1 * self.out_features, dim2)
        affine = affine.reshape(batch_size, len1 * self.out_features, -1)

        # (bz, len1 * out_features, dim2) * (bz, dim2, len2)
        # -> (bz, len1 * out_features, len2) -> (bz, len2, len1 * out_features)
        biaffine = torch.bmm(affine, input2.transpose(1, 2)).transpose(1, 2)

        # (bz, len2, len1, out_features)    # out_features: 1 or rel_size
        biaffine = biaffine.reshape((batch_size, len2, len1, -1))

        return biaffine