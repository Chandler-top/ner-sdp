from typing import Tuple
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from overrides import overrides

class BiLSTMEncoder(nn.Module):
    """
    BILSTM encoder.
    output the score of all labels.
    """

    def __init__(self, label_size: int, input_dim:int,
                 hidden_dim: int,
                 drop_lstm:float=0.5,
                 num_lstm_layers: int =1):
        super(BiLSTMEncoder, self).__init__()

        self.label_size = label_size
        print("[Model Info] Input size to LSTM: {}".format(input_dim))
        print("[Model Info] LSTM Hidden Size: {}".format(hidden_dim))
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_lstm_layers, batch_first=True, bidirectional=True)
        self.drop_lstm = nn.Dropout(drop_lstm)

        # 2022-01-06下面这句屏蔽，其实是linear_encoder.py的功能，放到transformers_nersdp.py中TransformersCRF()
        # self.hidden2tag = nn.Linear(hidden_dim, self.label_size)
    @overrides
    def forward(self, word_rep: torch.Tensor, word_seq_lens: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Encoding the input with BiLSTM
        :param word_rep: (batch_size, sent_len, input rep size)
        :param word_seq_lens: (batch_size, 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """
        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len.cpu(), True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_lstm(lstm_out)
        return feature_out, recover_idx
        # 2022-01-06下面这句屏蔽，其实是linear_encoder.py的功能，放到transformers_nersdp.py中TransformersCRF()
        # outputs = self.hidden2tag(feature_out)
        # return outputs[recover_idx]

# class BiLSTMEncoder(nn.Module):
#     def __init__(self, args):
#         super(BiLSTMEncoder, self).__init__()
#
#         self.bilstm = nn.LSTM(input_size=args.wd_embed_dim + args.tag_embed_dim,
#                               hidden_size=args.hidden_size // 2,
#                               num_layers=args.lstm_depth,
#                               dropout=args.lstm_drop,
#                               batch_first=True,
#                               bidirectional=True)
#
#     def forward(self, embed_inputs, non_pad_mask=None):
#         '''
#         :param embed_inputs: (bz, seq_len, embed_dim)
#         :param non_pad_mask: (bz, seq_len)
#         :return:
#         '''
#         if non_pad_mask is None:
#             non_pad_mask = embed_inputs.data.new_full(embed_inputs.shape[:2], 1)
#
#         seq_lens = non_pad_mask.data.sum(dim=1)
#         sort_lens, sort_idxs = torch.sort(seq_lens, dim=0, descending=True)
#         pack_embed = pack_padded_sequence(embed_inputs[sort_idxs], lengths=sort_lens, batch_first=True)
#         pack_enc_out, _ = self.bilstm(pack_embed)
#         enc_out, _ = pad_packed_sequence(pack_enc_out, batch_first=True)
#         _, unsort_idxs = torch.sort(sort_idxs, dim=0, descending=False)
#
#         return enc_out[unsort_idxs]

