# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelSmoothingLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1).cuda()
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='none').view(batch_size, num_pos, -1).sum(2)


def build_mask_matrix(seqlen, valid_len_list, prefix_len = 0):
    '''
        prefix_len: the length of prefix that we do not want to compute CL loss for.

        (1) if a sequence of length 4 contains zero padding token (i.e., the valid length is 4),
            then the loss padding matrix looks like
                 [0., 1., 1., 1.],
                 [1., 0., 1., 1.],
                 [1., 1., 0., 1.],
                 [1., 1., 1., 0.]

        (2) if a sequence of length 4 contains 1 padding token (i.e., the valid length is 3),
            then the loss padding matrix looks like
                 [0., 1., 1., 0.],
                 [1., 0., 1., 0.],
                 [1., 1., 0., 0.],
                 [0., 0., 0., 0.]
    '''
    res_list = []
    base_mask = torch.ones(seqlen, seqlen) - torch.eye(seqlen, seqlen)
    base_mask = base_mask.type(torch.FloatTensor)
    bsz = len(valid_len_list)
    for i in range(bsz):
        one_base_mask = base_mask.clone()
        one_valid_len = valid_len_list[i]
        one_base_mask[:,one_valid_len:] = 0.
        one_base_mask[one_valid_len:, :] = 0.
        if prefix_len > 0:
            one_base_mask[:prefix_len, :prefix_len] = 0.
        res_list.append(one_base_mask)
    res_mask = torch.stack(res_list, dim = 0)#torch.FloatTensor(res_list)
    #print (res_mask)
    assert res_mask.size() == torch.Size([bsz, seqlen, seqlen])
    return res_mask


def contrastive_loss(margin, score_matrix, input_ids, pad_token_id, prefix_len=0):
    '''
       margin: predefined margin to push similarity score away
       score_matrix: bsz x seqlen x seqlen
       input_ids: bsz x seqlen
       pad_token_id: indicating which tokens are padding token
    '''
    bsz, seqlen, _ = score_matrix.size()
    gold_score = torch.diagonal(score_matrix, offset=0, dim1=1, dim2=2) # bsz x seqlen
    gold_score = torch.unsqueeze(gold_score, -1)
    assert gold_score.size() == torch.Size([bsz, seqlen, 1])
    difference_matrix = gold_score - score_matrix
    assert difference_matrix.size() == torch.Size([bsz, seqlen, seqlen])
    loss_matrix = margin - difference_matrix # bsz x seqlen x seqlen
    loss_matrix = torch.nn.functional.relu(loss_matrix)

    ### input mask
    input_mask = torch.ones_like(input_ids).type(torch.FloatTensor)
    if loss_matrix.is_cuda:
        input_mask = input_mask.cuda(loss_matrix.get_device())
    input_mask = input_mask.masked_fill(input_ids.eq(pad_token_id), 0.0)

    if loss_matrix.is_cuda:
        input_mask = input_mask.cuda(loss_matrix.get_device())

    valid_len_list = torch.sum(input_mask, dim = -1).tolist()
    loss_mask = build_mask_matrix(seqlen, [int(item) for item in valid_len_list], prefix_len)
    if score_matrix.is_cuda:
        loss_mask = loss_mask.cuda(score_matrix.get_device())
    masked_loss_matrix = loss_matrix * loss_mask
    loss_matrix = torch.sum(masked_loss_matrix, dim = -1)
    loss_matrix = loss_matrix[:, :32]
    assert loss_matrix.size() == input_ids.size()
    loss_matrix = loss_matrix * input_mask
    cl_loss = torch.sum(loss_matrix) / torch.sum(loss_mask)
    return cl_loss
