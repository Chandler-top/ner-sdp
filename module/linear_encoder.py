
import torch
import torch.nn as nn


from overrides import overrides

class LinearEncoder(nn.Module):

    def __init__(self, label_size:int, hidden_dim:int):
        super(LinearEncoder, self).__init__()

        self.hidden2tag = nn.Linear(hidden_dim, label_size)

    @overrides
    def forward(self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor,
                lstm_feature: torch.Tensor, recover_idx:int) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        outputs = self.hidden2tag(lstm_feature)
        return outputs[recover_idx]


