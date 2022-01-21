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
import torch.nn.functional as F

from data_utils import START_TAG, STOP_TAG, PAD

from module.biaffine import *

def drop_sequence_sharedmask(inputs, dropout, batch_first=True):
    if batch_first:
        inputs = inputs.transpose(0, 1)
    seq_length, batch_size, hidden_size = inputs.size()
    drop_masks = inputs.data.new(batch_size, hidden_size).fill_(1 - dropout)
    drop_masks = Variable(torch.bernoulli(drop_masks), requires_grad=False)
    drop_masks = drop_masks / (1 - dropout)
    drop_masks = torch.unsqueeze(drop_masks, dim=2).expand(-1, -1, seq_length).permute(2, 0, 1)
    inputs = inputs * drop_masks

    return inputs.transpose(1, 0)

def pad_sequence(xs, length=None, padding=-1, dtype=np.float64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return torch.from_numpy(y)

def _model_var(model, x):
    p = next(filter(lambda p: p.requires_grad, model.parameters()))
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

class TransformersCRF(nn.Module):

    def __init__(self, config):
        super(TransformersCRF, self).__init__()
        self.device = config.device
        # Embeddings
        self.embedder = TransformersEmbedder(transformer_model_name=config.embedder_type,
                                             parallel_embedder=config.parallel_embedder)
        # BiLSTMEncoder or LinearEncoder
        self.dropout_mlp = config.dropout_mlp_hidden
        if config.hidden_dim > 0:
            self.lstmencoder = BiLSTMEncoder(label_size=config.label_size, input_dim=self.embedder.get_output_dim(),
                                         hidden_dim=config.hidden_dim, drop_lstm=config.dropout)

        # MLPs layer
        # self._activation = nn.ReLU()
        # self._activation = nn.ELU()
        # self._activation = nn.LeakyReLU(0.1)

        self.mlp_arc_dep = NonLinear(
            input_size = config.hidden_dim,
            hidden_size=config.arc_hidden_dim + config.rel_hidden_dim,
            activation=nn.LeakyReLU(0.1))
        self.mlp_arc_head = NonLinear(
            input_size= config.hidden_dim,
            hidden_size=config.arc_hidden_dim + config.rel_hidden_dim,
            activation=nn.LeakyReLU(0.1))

        self.total_num = int((config.arc_hidden_dim + config.rel_hidden_dim) / 100)
        self.arc_num = int(config.arc_hidden_dim / 100)
        self.rel_num = int(config.rel_hidden_dim / 100)

        self.arc_biaffine = Biaffine(config.arc_hidden_dim, config.arc_hidden_dim, \
                                     1, bias=(True, False))
        self.rel_biaffine = Biaffine(config.rel_hidden_dim, config.rel_hidden_dim, \
                                     config.rel_size, bias=(True, True))

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
                    synhead_ids: torch.Tensor, synlabel_ids: torch.Tensor,
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
        # lstm_scores = self.encoder(word_rep, word_seq_lens)
        lstm_feature, recover_idx = self.lstmencoder(word_rep, word_seq_lens)
        self.lstm_enc_hidden = lstm_feature
        lstm_scores = self.linencoder(word_rep, word_seq_lens, lstm_feature, recover_idx)
        # sdp-forward
        if self.training:
            lstm_outputs = drop_sequence_sharedmask(lstm_feature, self.dropout_mlp)#6*73*800
        x_all_dep = self.mlp_arc_dep(lstm_outputs)
        x_all_head = self.mlp_arc_head(lstm_outputs)

        if self.training:
            x_all_dep = drop_sequence_sharedmask(x_all_dep, self.dropout_mlp)
            x_all_head = drop_sequence_sharedmask(x_all_head, self.dropout_mlp)

        x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        x_all_head_splits = torch.split(x_all_head, 100, dim=2)

        x_arc_dep = torch.cat(x_all_dep_splits[:self.arc_num], dim=2)
        x_arc_head = torch.cat(x_all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(x_arc_dep, x_arc_head)
        arc_logit = torch.squeeze(arc_logit, dim=3)

        x_rel_dep = torch.cat(x_all_dep_splits[self.arc_num:], dim=2)
        x_rel_head = torch.cat(x_all_head_splits[self.arc_num:], dim=2)

        rel_logit = self.rel_biaffine(x_rel_dep, x_rel_head)
        # cache
        self.arc_logits = arc_logit
        self.rel_logits = rel_logit
        # sdp loss
        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)

        max_seq_len = max(word_seq_lens).item()
        non_pad_mask = torch.zeros((batch_size, max_seq_len)).to(self.device)
        non_pad_mask = non_pad_mask.byte()
        num_rows, num_cols = synhead_ids.shape[0], synhead_ids.shape[1]
        for i in range(num_rows):
            a = synhead_ids[i, :]
            non_pad_mask[i, :word_seq_lens[i].item()].fill_(1)

        sdp_loss, arc_acc, rel_acc, total_arcs = self.cal_sdp_loss(arc_logit,rel_logit,synhead_ids,synlabel_ids,non_pad_mask)

        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev).view(1, sent_len).expand(batch_size, sent_len)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
        unlabed_score, labeled_score =  self.inferencer(lstm_scores, word_seq_lens, labels, mask)
        return unlabed_score - labeled_score, sdp_loss, arc_acc, rel_acc, total_arcs

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
        # 这个地方加入 sdp 的 解析
        features = self.linencoder(word_rep, word_seq_lens, lstm_features, recover_idx)
        bestScores, decodeIdx = self.inferencer.decode(features, word_seq_lens)
        return bestScores, decodeIdx

    def cal_sdp_loss(self, pred_arcs, pred_rels, true_arcs, true_rels, non_pad_mask):
        '''
        :param pred_arcs: (bz, seq_len, seq_len) 预测已经带了padding
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_arcs: (bz, seq_len)  包含padding -1 已经在transformers_dataset.py 的collate_fn中完成
        :param true_rels: (bz, seq_len) 同上
        :param non_pad_mask: (bz, seq_len) 有效部分mask
        :return:
        '''
        pad_mask = (non_pad_mask == 0)

        bz, seq_len, _ = pred_arcs.size()
        masked_true_heads = true_arcs.masked_fill(pad_mask, -100)
        arc_loss = F.cross_entropy(pred_arcs.reshape(bz * seq_len, -1),
                                   masked_true_heads.reshape(-1))
        # print("arc_loss:", arc_loss)
        # return arc_loss
        bz, seq_len, seq_len, rel_size = pred_rels.size()

        out_rels = pred_rels[torch.arange(bz, device=pred_arcs.device, dtype=torch.long).unsqueeze(1),
                             torch.arange(seq_len, device=pred_arcs.device, dtype=torch.long).unsqueeze(0),
                             true_arcs].contiguous()

        masked_true_rels = true_rels.masked_fill(pad_mask, -100)

        # (bz*seq_len, rel_size)  (bz*seq_len, )
        rel_loss = F.cross_entropy(out_rels.reshape(-1, rel_size),
                                   masked_true_rels.reshape(-1))

        arc_acc, rel_acc, total_arcs = self.calc_sdp_acc(pred_arcs, pred_rels, true_arcs, true_rels, non_pad_mask)
        return arc_loss + rel_loss, arc_acc, rel_acc, total_arcs

    def calc_sdp_acc(self, pred_arcs, pred_rels, true_heads, true_rels, non_pad_mask=None):
        '''a
        :param pred_arcs: (bz, seq_len, seq_len)
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_heads: (bz, seq_len)  包含padding
        :param true_rels: (bz, seq_len)
        :param non_pad_mask: (bz, seq_len) 非填充部分mask
        :return:
        '''
        # non_pad_mask[:, 0] = 0  # mask out <root>
        _mask = non_pad_mask.bool()

        bz, seq_len, seq_len, rel_size = pred_rels.size()

        # (bz, seq_len)
        pred_heads = pred_arcs.data.argmax(dim=2)
        masked_pred_heads = pred_heads[_mask]
        masked_true_heads = true_heads[_mask]
        arc_acc = masked_true_heads.eq(masked_pred_heads).sum().item()

        total_arcs = non_pad_mask.sum().item()

        out_rels = pred_rels[torch.arange(bz, device=pred_arcs.device, dtype=torch.long).unsqueeze(1),
                             torch.arange(seq_len, device=pred_arcs.device, dtype=torch.long).unsqueeze(0),
                             true_heads].contiguous()
        pred_rels = out_rels.argmax(dim=2)
        masked_pred_rels = pred_rels[_mask]
        masked_true_rels = true_rels[_mask]
        rel_acc = masked_true_rels.eq(masked_pred_rels).sum().item()

        return arc_acc, rel_acc, total_arcs