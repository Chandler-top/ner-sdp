#
# @author: Allan
#

import torch
import torch.nn as nn

from module.bilstm_encoder import BiLSTMEncoder
from module.linear_crf_inferencer import LinearCRF
from module.linear_encoder import LinearEncoder
from embedder.transformers_embedder import TransformersEmbedder
from typing import Tuple
from overrides import overrides

from data_utils import START_TAG, STOP_TAG, PAD

from module.biaffine import *

class TransformersCRF(nn.Module):

    def __init__(self, config):
        super(TransformersCRF, self).__init__()
        # Embeddings
        self.embedder = TransformersEmbedder(transformer_model_name=config.embedder_type,
                                             parallel_embedder=config.parallel_embedder)
        # BiLSTMEncoder or LinearEncoder
        self.dropout_mlp = config.dropout_mlp_hidden
        if config.hidden_dim > 0:
            self.lstmencoder = BiLSTMEncoder(label_size=config.label_size, input_dim=self.embedder.get_output_dim(),
                                         hidden_dim=config.hidden_dim, drop_lstm=config.dropout)

        # # MLPs layer
        # self._activation = nn.ReLU()
        # # self._activation = nn.ELU()
        # # self._activation = nn.LeakyReLU(0.1)
        #
        # self.mlp_arc_dep = NonLinear(
        #     input_size=2 * config.lstm_hiddens,
        #     hidden_size=config.mlp_arc_size + config.mlp_rel_size,
        #     activation=nn.LeakyReLU(0.1))
        # self.mlp_arc_head = NonLinear(
        #     input_size=2 * config.lstm_hiddens,
        #     hidden_size=config.mlp_arc_size + config.mlp_rel_size,
        #     activation=nn.LeakyReLU(0.1))
        #
        # self.total_num = int((config.mlp_arc_size + config.mlp_rel_size) / 100)
        # self.arc_num = int(config.mlp_arc_size / 100)
        # self.rel_num = int(config.mlp_rel_size / 100)
        #
        # self.arc_biaffine = Biaffine(config.mlp_arc_size, config.mlp_arc_size, \
        #                              1, bias=(True, False))
        # self.rel_biaffine = Biaffine(config.mlp_rel_size, config.mlp_rel_size, \
        #                              vocab.rel_size, bias=(True, True))

        # else:
        self.linencoder = LinearEncoder(label_size=config.label_size, hidden_dim=config.hidden_dim)



        # CRF
        self.inferencer = LinearCRF(label_size=config.label_size, label2idx=config.label2idx, add_iobes_constraint=config.add_iobes_constraint,
                                    idx2labels=config.idx2labels)
        self.pad_idx = config.label2idx[PAD]


    @overrides
    def forward(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    orig_to_tok_index: torch.Tensor,
                    input_mask: torch.Tensor,
                    labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate the negative loglikelihood.
        :param words: (batch_size x max_seq_len)
        :param word_seq_lens: (batch_size)
        :param context_emb: (batch_size x max_seq_len x context_emb_size)
        :param chars: (batch_size x max_seq_len x max_char_len)
        :param char_seq_lens: (batch_size x max_seq_len)
        :param labels: (batch_size x max_seq_len)
        :return: the total negative log-likelihood loss
        """
        word_rep = self.embedder(words, orig_to_tok_index, input_mask)
        # 2022-01-06
        # lstm_scores = self.encoder(word_rep, word_seq_lens)
        lstm_feature, recover_idx = self.lstmencoder(word_rep, word_seq_lens)
        lstm_scores = self.linencoder(word_rep, word_seq_lens, lstm_feature, recover_idx)
        # sdp-2022-01-04
        # x_all_dep = self.mlp_arc_dep(outputs)
        # x_all_head = self.mlp_arc_head(outputs)
        #
        # x_all_dep = drop_sequence_sharedmask(x_all_dep, self.dropout_mlp)
        # x_all_head = drop_sequence_sharedmask(x_all_head, self.dropout_mlp)
        #
        # x_all_dep_splits = torch.split(x_all_dep, split_size=100, dim=2)
        # x_all_head_splits = torch.split(x_all_head, split_size=100, dim=2)
        #
        # x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        # x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)
        #
        # arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        # arc_logit = torch.squeeze(arc_logit, dim=3)
        #
        # x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        # x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)
        #
        # rel_logit_cond = self.rel_biaffine(x_rel_dep, x_rel_head)
        # return arc_logit, rel_logit_cond

        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)
        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev).view(1, sent_len).expand(batch_size, sent_len)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
        unlabed_score, labeled_score =  self.inferencer(lstm_scores, word_seq_lens, labels, mask)
        return unlabed_score - labeled_score

    def decode(self, words: torch.Tensor,
                    word_seq_lens: torch.Tensor,
                    orig_to_tok_index: torch.Tensor,
                    input_mask,
                    **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        word_rep = self.embedder(words, orig_to_tok_index, input_mask)
        lstm_features, recover_idx = self.lstmencoder(word_rep, word_seq_lens)
        features = self.linencoder(word_rep, word_seq_lens, lstm_features, recover_idx)
        bestScores, decodeIdx = self.inferencer.decode(features, word_seq_lens)
        return bestScores, decodeIdx