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
        self._activation = nn.LeakyReLU(0.1)

        # self.mlp_arc_dep = NonLinear(
        #     input_size = config.hidden_dim,
        #     hidden_size=config.arc_hidden_dim + config.rel_hidden_dim,
        #     activation=nn.LeakyReLU(0.1))
        # self.mlp_arc_head = NonLinear(
        #     input_size= config.hidden_dim,
        #     hidden_size=config.arc_hidden_dim + config.rel_hidden_dim,
        #     activation=nn.LeakyReLU(0.1))
        #
        # self.total_num = int((config.arc_hidden_dim + config.rel_hidden_dim) / 100)
        # self.arc_num = int(config.arc_hidden_dim / 100)
        # self.rel_num = int(config.rel_hidden_dim / 100)
        #
        # self.arc_biaffine = Biaffine(config.arc_hidden_dim, config.arc_hidden_dim, \
        #                              1, bias=(True, False))
        # self.rel_biaffine = Biaffine(config.rel_hidden_dim, config.rel_hidden_dim, \
        #                              config.rel_size, bias=(True, True))
        #
        self.linencoder = LinearEncoder(label_size=config.label_size, hidden_dim=config.hidden_dim)

        self.mlp_arc = LindgeW_NonlinearMLP(in_feature=config.hidden_dim,
                                    out_feature=config.arc_hidden_dim * 2,
                                    activation=self._activation)
        self.mlp_lbl = LindgeW_NonlinearMLP(in_feature=config.hidden_dim,
                                    out_feature=config.rel_hidden_dim * 2,
                                    activation=self._activation)

        self.arc_biaffine = LindgeW_Biaffine(config.arc_hidden_dim,
                                     1, bias=(True, False))

        self.label_biaffine = LindgeW_Biaffine(config.rel_hidden_dim,
                                       config.rel_size, bias=(True, True))
        # CRF
        self.inferencer = LinearCRF(label_size=config.label_size, label2idx=config.label2idx, add_iobes_constraint=config.add_iobes_constraint,
                                    idx2labels=config.idx2labels)
        self.pad_idx = config.label2idx[PAD]


    @overrides
    def forward(self, iter: int, words: torch.Tensor,
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
        # if self.training:
        #     lstm_outputs = drop_sequence_sharedmask(lstm_feature, self.dropout_mlp)#6*73*800
        # x_all_dep = self.mlp_arc_dep(lstm_outputs)
        # x_all_head = self.mlp_arc_head(lstm_outputs)
        #
        # if self.training:
        #     x_all_dep = drop_sequence_sharedmask(x_all_dep, self.dropout_mlp)
        #     x_all_head = drop_sequence_sharedmask(x_all_head, self.dropout_mlp)
        #
        # x_all_dep_splits = torch.split(x_all_dep, 100, dim=2)
        # x_all_head_splits = torch.split(x_all_head, 100, dim=2)
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
        # rel_logit = self.rel_biaffine(x_rel_dep, x_rel_head)
        if self.training:
            enc_out = timestep_dropout(lstm_feature, self.dropout_mlp)

        arc_feat = self.mlp_arc(enc_out)
        lbl_feat = self.mlp_lbl(enc_out)
        arc_head, arc_dep = arc_feat.chunk(2, dim=-1)
        lbl_head, lbl_dep = lbl_feat.chunk(2, dim=-1)

        if self.training:
            arc_head = timestep_dropout(arc_head, self.dropout_mlp)
            arc_dep = timestep_dropout(arc_dep, self.dropout_mlp)
        arc_logit = self.arc_biaffine(arc_dep, arc_head).squeeze(-1)

        if self.training:
            lbl_head = timestep_dropout(lbl_head, self.dropout_mlp)
            lbl_dep = timestep_dropout(lbl_dep, self.dropout_mlp)
        rel_logit = self.label_biaffine(lbl_dep, lbl_head)
        # cache
        self.arc_logits = arc_logit
        self.rel_logits = rel_logit
        # sdp loss
        # lengths = max(word_seq_lens).item()
        # lengths = word_seq_lens.tolist()
        batch_size = word_rep.size(0)
        sent_len = word_rep.size(1)
        true_arcs, true_rels, no_pad_mask = self.compute_true_arc_rel(synhead_ids, synlabel_ids,
                                                word_seq_lens, batch_size)

        sdp_loss = self.cal_sdp_loss(arc_logit,rel_logit,synhead_ids,synlabel_ids,no_pad_mask)
        # sdp_loss = self.compute_sdp_loss(true_arcs,true_rels,lengths)

        dev_num = word_rep.get_device()
        curr_dev = torch.device(f"cuda:{dev_num}") if dev_num >= 0 else torch.device("cpu")
        maskTemp = torch.arange(1, sent_len + 1, dtype=torch.long, device=curr_dev).view(1, sent_len).expand(batch_size, sent_len)
        mask = torch.le(maskTemp, word_seq_lens.view(batch_size, 1).expand(batch_size, sent_len))
        unlabed_score, labeled_score =  self.inferencer(lstm_scores, word_seq_lens, labels, mask)
        ner_loss = unlabed_score - labeled_score
        return ner_loss

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

    def compute_sdp_loss(self, true_arcs, true_rels, lengths):
        b, l1, l2 = self.arc_logits.size()
        x = pad_sequence(true_arcs, length=l1, padding=0, dtype=np.int64)
        index_true_arcs = torch.autograd.Variable(x)
        x = pad_sequence(true_arcs, length=l1, padding=-1, dtype=np.int64)
        x = x.to(self.device)
        true_arcs = torch.autograd.Variable(x)

        masks = []
        for length in lengths:
            mask = torch.FloatTensor([0] * length + [-10000] * (l2 - length))
            mask = torch.autograd.Variable(mask)
            mask = torch.unsqueeze(mask, dim=1).expand(-1, l1)
            masks.append(mask.transpose(0, 1))
        length_mask = torch.stack(masks, 0)
        arc_logits = self.arc_logits + length_mask.to(self.device)

        arc_loss = F.cross_entropy(
            arc_logits.view(b * l1, l2), true_arcs.view(b * l1),
            ignore_index=-1)

        size = self.rel_logits.size()
        # output_logits = _model_var(self.model, torch.zeros(size[0], size[1], size[3]))
        x = torch.zeros(size[0], size[1], size[3])
        x = x.to(self.device)
        output_logits = torch.autograd.Variable(x)

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][int(arcs[i])])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        b, l1, d = output_logits.size()
        # true_rels = _model_var(self.model, pad_sequence(true_rels, padding=-1, dtype=np.int64))
        x = pad_sequence(true_rels, padding=-1, dtype=np.int64)
        x = x.to(self.device)
        true_rels = torch.autograd.Variable(x)

        rel_loss = F.cross_entropy(
            output_logits.view(b * l1, d), true_rels.view(b * l1), ignore_index=-1)
        loss = arc_loss + rel_loss
        return loss

    def compute_sdp_accuracy(self, true_arcs, true_rels):
        b, l1, l2 = self.arc_logits.size()
        pred_arcs = self.arc_logits.data.max(2)[1].cpu()
        index_true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        arc_correct = pred_arcs.eq(true_arcs).cpu().sum()


        size = self.rel_logits.size()
        output_logits = _model_var(self.model, torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][arcs[i]])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        pred_rels = output_logits.data.max(2)[1].cpu()
        true_rels = pad_sequence(true_rels, padding=-1, dtype=np.int64)
        label_correct = pred_rels.eq(true_rels).cpu().sum()

        total_arcs = b * l1 - np.sum(true_arcs.cpu().numpy() == -1)

        return arc_correct, label_correct, total_arcs

    def compute_true_arc_rel(self, tensor_arcs, tensor_rels, word_seq_len, batch_size):
        true_arcs = []
        true_rels = []
        max_seq_len = max(word_seq_len).item()
        # true_arcs = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        # true_rels = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

        non_pad_mask = torch.zeros((batch_size, max_seq_len))
        non_pad_mask = non_pad_mask.byte()

        num_rows, num_cols = tensor_arcs.shape[0], tensor_arcs.shape[1]
        for i in range(num_rows):
            a = tensor_arcs[i, :]
            true_arcs.append(a[0:word_seq_len[i].item()].numpy())
            non_pad_mask[i, :word_seq_len[i].item()].fill_(1)

        num_rows, num_cols = tensor_rels.shape[0], tensor_rels.shape[1]
        for i in range(num_rows):
            a = tensor_rels[i, :]
            true_rels.append(a[0:word_seq_len[i].item()].numpy())

        return true_arcs, true_rels, non_pad_mask

    def cal_sdp_loss(self, pred_arcs, pred_rels, true_arcs, true_rels, non_pad_mask):
        '''
        :param pred_arcs: (bz, seq_len, seq_len) 预测已经带了padding
        :param pred_rels:  (bz, seq_len, seq_len, rel_size)
        :param true_arcs: (bz, seq_len)  包含padding -100 已经在transformers_dataset.py 的collate_fn中完成
        :param true_rels: (bz, seq_len) 同上
        :param non_pad_mask: (bz, seq_len) 有效部分mask
        :return:
        '''
        pad_mask = (non_pad_mask == 0)

        bz, seq_len, _ = pred_arcs.size()
        # print("pred_arcs_size:", bz, seq_len, _)
        # print("true_arcs_size:",true_arcs.size())
        masked_true_heads = true_arcs.masked_fill(pad_mask, -1).to(self.device)
        arc_loss = F.cross_entropy(pred_arcs.reshape(bz*seq_len, -1),
                                   masked_true_heads.reshape(-1),
                                   ignore_index=-1)

        bz, seq_len, seq_len, rel_size = pred_rels.size()
        # print ("pred_rels_size:",pred_rels.size())
        # print("true_rels_size:", true_rels.size())

        out_rels = pred_rels[torch.arange(bz, device=pred_arcs.device, dtype=torch.long).unsqueeze(1),
                             torch.arange(seq_len, device=pred_arcs.device, dtype=torch.long).unsqueeze(0),
                             true_arcs].contiguous()
        return arc_loss
        masked_true_rels = true_rels.masked_fill(pad_mask, -1).to(self.device)
        # (bz*seq_len, rel_size)  (bz*seq_len, )
        rel_loss = F.cross_entropy(out_rels.reshape(-1, rel_size),
                                   masked_true_rels.reshape(-1),
                                   ignore_index=-1)
        # print("rel_loss:", rel_loss)
        return arc_loss + rel_loss
        
