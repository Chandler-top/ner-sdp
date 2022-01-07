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

        # MLPs layer
        # self._activation = nn.ReLU()
        # self._activation = nn.ELU()
        self._activation = nn.LeakyReLU(0.1)
        self.mlp_arc = NonlinearMLP(in_feature=config.hidden_dim,
                                    out_feature=config.arc_hidden_dim * 2,
                                    # out_feature=config.arc_hidden_dim + config.rel_hidden_dim
                                    activation=self._activation)
        self.mlp_lbl = NonlinearMLP(in_feature=config.hidden_dim,
                                    out_feature=config.rel_hidden_dim * 2,
                                    # out_feature=config.mlp_arc_size+config.mlp_rel_size
                                    activation=self._activation)

        self.arc_biaffine = Biaffine(config.arc_hidden_dim,
                                     1, bias=(True, False))

        self.label_biaffine = Biaffine(config.rel_hidden_dim,
                                       config.rel_size, bias=(True, True))

        # self.word_fc = nn.Linear(args.wd_embed_dim, args.wd_embed_dim)
        # self.tag_fc = nn.Linear(args.tag_embed_dim, args.tag_embed_dim)
        # self.word_norm = nn.LayerNorm(args.wd_embed_dim)
        # self.tag_norm = nn.LayerNorm(args.tag_embed_dim)
        # self.reset_parameters()

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
        self.lstm_enc_hidden = lstm_feature
        lstm_scores = self.linencoder(word_rep, word_seq_lens, lstm_feature, recover_idx)
        # sdp-2022-01-04
        # enc dropout 已经有了所以屏蔽了，后面根据效果确定是否打开
        # if self.training:
        #     enc_out = timestep_dropout(lstm_feature, config.dropout)

        arc_feat = self.mlp_arc(lstm_feature)
        lbl_feat = self.mlp_lbl(lstm_feature)
        arc_head, arc_dep = arc_feat.chunk(2, dim=-1)
        lbl_head, lbl_dep = lbl_feat.chunk(2, dim=-1)

        if self.training:
            arc_head = timestep_dropout(arc_head, self.dropout_mlp)
            arc_dep = timestep_dropout(arc_dep, self.dropout_mlp)
        arc_score = self.arc_biaffine(arc_dep, arc_head).squeeze(-1)

        if self.training:
            lbl_head = timestep_dropout(lbl_head, self.dropout_mlp)
            lbl_dep = timestep_dropout(lbl_dep, self.dropout_mlp)
        lbl_score = self.label_biaffine(lbl_dep, lbl_head)

        # return arc_score, lbl_score
        # crf
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

    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.embedder.weight)
    #     nn.init.xavier_uniform_(self.embedder.weight)

    def compute_sdp_loss(self, true_arcs, true_rels, lengths):
        b, l1, l2 = self.arc_logits.size()
        index_true_arcs = _model_var(
            self.model,
            pad_sequence(true_arcs, length=l1, padding=0, dtype=np.int64))
        true_arcs = _model_var(
            self.model,
            pad_sequence(true_arcs, length=l1, padding=-1, dtype=np.int64))

        masks = []
        for length in lengths:
            mask = torch.FloatTensor([0] * length + [-10000] * (l2 - length))
            mask = _model_var(self.model, mask)
            mask = torch.unsqueeze(mask, dim=1).expand(-1, l1)
            masks.append(mask.transpose(0, 1))
        length_mask = torch.stack(masks, 0)
        arc_logits = self.arc_logits + length_mask

        arc_loss = F.cross_entropy(
            arc_logits.view(b * l1, l2), true_arcs.view(b * l1),
            ignore_index=-1)

        size = self.rel_logits.size()
        output_logits = _model_var(self.model, torch.zeros(size[0], size[1], size[3]))

        for batch_index, (logits, arcs) in enumerate(zip(self.rel_logits, index_true_arcs)):
            rel_probs = []
            for i in range(l1):
                rel_probs.append(logits[i][int(arcs[i])])
            rel_probs = torch.stack(rel_probs, dim=0)
            output_logits[batch_index] = torch.squeeze(rel_probs, dim=1)

        b, l1, d = output_logits.size()
        true_rels = _model_var(self.model, pad_sequence(true_rels, padding=-1, dtype=np.int64))

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