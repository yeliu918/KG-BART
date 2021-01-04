from random import randint, shuffle, choice
from random import random as rand
import math
import torch

from KGBART_training.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline
import json
from random import sample, randint, random
from tqdm import tqdm


# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None,
                         always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b


class Seq2SeqDataset(torch.utils.data.Dataset):
    """ Load sentence pair (sequential or random order) from corpus """

    def __init__(self, file_src, file_tgt, file_entity_id, file_relation_id, file_onehop, batch_size, tokenizer,
                 max_len, pretraining_KG=None, pretraining_num = 200000,
                 short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[]):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.pretraining_num = pretraining_num
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order

        # read the file into memory
        self.ex_list = []

        if pretraining_KG is None:
            with open(file_src, "r", encoding='utf-8') as f_src, open(file_tgt, "r", encoding='utf-8') as f_tgt, \
                    open(file_entity_id, "r", encoding='utf-8') as f_entity, open(file_relation_id, "r",
                                                                                  encoding='utf-8') as f_relation, \
                    open(file_onehop, "r", encoding='utf-8') as f_onehop:
                entity_id = {}
                id_entity = {}
                entity_line = f_entity.readlines()

                for i, token in enumerate(["</s>", "<s>", "<pad>", "<mask>"]):
                    entity_id[token] = i
                    id_entity[i] = token
                for i, entityid in enumerate(entity_line):
                    if i == 0: continue
                    entity_id[entityid.split()[0]] = int(entityid.split()[1]) + 4
                    id_entity[int(entityid.split()[1]) + 4] = entityid.split()[0]

                relation_id = {}
                id_relation = {}
                relation_line = f_relation.readlines()
                for i, token in enumerate(["</s>", "<s>", "<pad>", "<mask>"]):
                    relation_id[token] = i
                    id_relation[i] = token
                for i, relationid in enumerate(relation_line):
                    if i == 0: continue
                    relation_id[relationid.split()[0]] = int(relationid.split()[1]) + 4
                    id_relation[int(relationid.split()[1])] = relationid.split()[0]

                concept_list = json.load(f_onehop)

                for src, tgt, onehop in zip(f_src, f_tgt, concept_list):
                    src_entity_id = []
                    src_tk = []
                    word_subword = []
                    for wi, src1 in enumerate(src.split()):
                        src_tk1 = tokenizer.tokenize(src1)
                        src_entity_id.append(entity_id[src1])
                        word_subword.append(len(src_tk1))
                        for tk in src_tk1:
                            src_tk.append(tk)

                    concept_entity_expand = []
                    concept_entity_mask = []
                    concept_relation_expand = []
                    concept_relation_mask = []
                    for key in onehop.keys():
                        each_concept_expand = [entity_id[key]]
                        each_relation_expand = [2]  ## no relation with itself, pad
                        for word_rel in onehop[key].keys():
                            if len(each_concept_expand) >= 5: break
                            ng_entity_id = entity_id[word_rel]
                            ng_relation_id = relation_id[onehop[key][word_rel][0]]
                            each_concept_expand.append(ng_entity_id)
                            each_relation_expand.append(ng_relation_id)
                        pad_len = 5 - len(each_concept_expand)
                        concept_entity_mask.append(len(each_concept_expand) * [1] + pad_len * [0])
                        concept_relation_mask.append(len(each_concept_expand) * [1] + pad_len * [0])

                        if len(each_concept_expand) < 5:
                            each_concept_expand.extend(pad_len * [2])
                            each_relation_expand.extend(pad_len * [2])
                        concept_entity_expand.append(each_concept_expand)
                        concept_relation_expand.append(each_relation_expand)

                    pad_len = 5 - len(concept_entity_expand)
                    for i in range(pad_len):
                        concept_entity_expand.append([2] * 5)
                        concept_entity_mask.append([0] * 5)
                        concept_relation_expand.append([2] * 5)
                        concept_relation_mask.append([0] * 5)

                    tgt_tk = tokenizer.tokenize(tgt.strip())
                    assert len(src_tk) > 0
                    assert len(tgt_tk) > 0
                    assert len(src_entity_id) == len(word_subword)

                    self.ex_list.append(
                        (src_tk, tgt_tk, src_entity_id, word_subword, concept_entity_expand, concept_relation_expand))

        else:
            with open(pretraining_KG, "r", encoding='utf-8') as f_entity:
                entity_line = f_entity.readlines()
                entity_id = {}
                id_entity = {}
                all_entity = []
                for i, token in enumerate(["</s>", "<s>", "<pad>", "<mask>"]):
                    entity_id[token] = i
                    id_entity[i] = token
                for i, entityid in enumerate(entity_line):
                    if i == 0: continue
                    all_entity.append(entityid.split()[0])
                    entity_id[entityid.split()[0]] = int(entityid.split()[1]) + 4
                    id_entity[int(entityid.split()[1]) + 4] = entityid.split()[0]

                while len(self.ex_list) < self.pretraining_num:
                    src = sample(all_entity, 5)
                    mask_num = randint(1, 5)
                    src_entity_id = []
                    word_subword = []
                    src_tk = []
                    tgt_tk = []
                    masked_num = 0
                    for e_i, entity in enumerate(src):
                        entity_tk = tokenizer.tokenize(entity)
                        src_entity_id.append(entity_id[entity])
                        word_subword.append(len(entity_tk))
                        if random() >= 0.5 and mask_num - masked_num > 0:
                            for tk in entity_tk:
                                src_tk.append("<mask>")
                                tgt_tk.append(tk)
                            masked_num += 1
                        else:
                            for tk in entity_tk:
                                src_tk.append(tk)
                                tgt_tk.append(tk)
                    self.ex_list.append((src_tk, tgt_tk, src_entity_id, word_subword))
        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choice(self.bi_uni_pipeline)
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list) - 1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


class Preprocess4Pretrain(Pipeline):
    def __init__(self, vocab_words, indexer, max_len=512, new_segment_ids=False, truncate_config={},
                 mask_source_words=False, mode="s2s", pretraining_KG=False, num_qkv=0, s2s_special_token=False,
                 s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3  # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.pretraining_KG = pretraining_KG
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

    def __call__(self, instance):
        tokens_a, tokens_b, entity_id_a, word_subword = instance
        sep_token = "</s>"
        cls_token = "<s>"
        pad_token = "<pad>"
        mask_token = "<mask>"

        tokens_a = [cls_token] + tokens_a + [sep_token]
        tokens_b = [cls_token] + tokens_b + [sep_token]

        n_pad = self.max_len_a - len(word_subword)
        word_subword.extend([0] * n_pad)

        labels = [t for t in tokens_b[1:]]
        if self.pos_shift:
            tokens_b = ['[S2S_SOS]'] + tokens_b

        input_entity_id = self.indexer([cls_token]) + entity_id_a + self.indexer([sep_token])
        input_ids = self.indexer(tokens_a)  # [self.indexer(tokens) for tokens in tokens_a]
        decoder_input_ids = self.indexer(tokens_b)  # [self.indexer(tokens) for tokens in tokens_b]

        # Zero Padding
        n_pad = self.max_len_a - len(input_ids)
        subword_mask = [1] * len(input_ids)
        subword_mask.extend([0] * n_pad)
        input_ids.extend([self.indexer(pad_token)] * n_pad)

        n_pad = self.max_len_a - len(input_entity_id)
        word_mask = [1] * len(input_entity_id)
        word_mask.extend([0] * n_pad)
        input_entity_id.extend([self.indexer(pad_token)] * n_pad)

        n_pad = self.max_len_b - len(decoder_input_ids)
        decoder_input_ids.extend([self.indexer(pad_token)] * n_pad)
        decoder_attention_mask = [1] * len(tokens_b)
        decoder_attention_mask.extend([0] * n_pad)
        labels = self.indexer(labels)  # [self.indexer(tokens) for tokens in labels]
        labels.extend([-100] * (n_pad + 1))

        if self.num_qkv > 1:
            mask_qkv = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            mask_qkv.extend([0] * n_pad)
        else:
            mask_qkv = None
        return (input_ids, input_entity_id, subword_mask, word_mask, word_subword, decoder_input_ids, decoder_attention_mask,
        labels)


class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, new_segment_ids=False, truncate_config={},
                 mask_source_words=False, mode="s2s", pretraining_KG=False, num_qkv=0, s2s_special_token=False,
                 s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_a = truncate_config.get('max_len_a', None)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.task_idx = 3  # relax projection layer for different tasks
        self.mask_source_words = mask_source_words
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.pretraining_KG = pretraining_KG
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

    def __call__(self, instance):
        tokens_a, tokens_b, entity_id_a, word_subword, concept_entity_expand, concept_relation_expand = instance
        sep_token = "</s>"
        cls_token = "<s>"
        pad_token = "<pad>"
        mask_token = "<mask>"

        tokens_a = [cls_token] + tokens_a + [sep_token]
        tokens_b = [cls_token] + tokens_b + [sep_token]

        n_pad = self.max_len_a - len(word_subword)
        word_subword.extend([0] * n_pad)

        labels = [t for t in tokens_b[1:]]
        if self.pos_shift:
            tokens_b = ['[S2S_SOS]'] + tokens_b

        input_entity_id = self.indexer([cls_token]) + entity_id_a + self.indexer([sep_token])
        input_ids = self.indexer(tokens_a)  # [self.indexer(tokens) for tokens in tokens_a]
        decoder_input_ids = self.indexer(tokens_b)  # [self.indexer(tokens) for tokens in tokens_b]

        # Zero Padding
        n_pad = self.max_len_a - len(input_ids)
        subword_mask = [1] * len(input_ids)
        subword_mask.extend([0] * n_pad)
        input_ids.extend([self.indexer(pad_token)] * n_pad)

        n_pad = self.max_len_a - len(input_entity_id)
        word_mask = [1] * len(input_entity_id)
        word_mask.extend([0] * n_pad)
        input_entity_id.extend([self.indexer(pad_token)] * n_pad)

        n_pad = self.max_len_b - len(decoder_input_ids)
        decoder_input_ids.extend([self.indexer(pad_token)] * n_pad)
        decoder_attention_mask = [1] * len(tokens_b)
        decoder_attention_mask.extend([0] * n_pad)
        labels = self.indexer(labels)  # [self.indexer(tokens) for tokens in labels]
        labels.extend([-100] * (n_pad + 1))

        if self.num_qkv > 1:
            mask_qkv = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
            mask_qkv.extend([0] * n_pad)
        else:
            mask_qkv = None

        return (input_ids, input_entity_id, subword_mask, word_mask, word_subword, decoder_input_ids, decoder_attention_mask,
        concept_entity_expand, concept_relation_expand, labels)


class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s",
                 num_qkv=0, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
        super().__init__()
        self.max_len = max_len
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.max_tgt_length = max_tgt_length
        self.num_qkv = num_qkv
        self.s2s_special_token = s2s_special_token
        self.s2s_add_segment = s2s_add_segment
        self.s2s_share_segment = s2s_share_segment
        self.pos_shift = pos_shift

    def __call__(self, instance):
        tokens_a, entity_id_a, word_subword, concept_entity_expand, concept_relation_expand = instance
        sep_token = "</s>"
        cls_token = "<s>"
        pad_token = "<pad>"
        mask_token = "<mask>"

        input_entity_id = self.indexer([cls_token]) + entity_id_a + self.indexer([sep_token])

        tokens_a = [cls_token] + tokens_a + [sep_token]

        n_pad = self.max_len - len(input_entity_id)
        word_mask = [1] * len(input_entity_id)
        word_mask.extend([0] * n_pad)
        input_entity_id.extend([self.indexer(pad_token)] * n_pad)

        n_pad = self.max_len - len(word_subword)
        word_subword.extend([0] * n_pad)

        tokens = tokens_a

        # Token Indexing
        input_ids = [self.indexer(t) for t in tokens]

        if len(input_ids) >= self.max_len:
            print(tokens)
        input_mask = [1] * len(tokens)

        n_pad = self.max_len - len(input_ids)
        subword_mask = [1] * len(input_ids)
        subword_mask.extend([0] * n_pad)
        input_ids.extend([self.indexer(pad_token)] * n_pad)
        input_mask.extend([0] * n_pad)

        assert len(input_ids) == self.max_len
        assert len(input_mask) == self.max_len
        assert len(input_entity_id) == self.max_len
        return (input_ids, input_entity_id, subword_mask, word_mask, word_subword, concept_entity_expand,
                concept_relation_expand)
