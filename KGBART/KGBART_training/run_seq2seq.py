"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import math
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import os
import sys

os.chdir("../")
PROJECT_DIR = os.getcwd()
print("Current Working Directory ", os.getcwd())
sys_path = os.path.join(os.getcwd(), 'KGBART')
sys.path.insert(1, sys_path)
from nn.data_parallel import DataParallelImbalance
import KGBART_training.seq2seq_loader as seq2seq_loader
from KGBART_model.tokenization_bart import MBartTokenizer, BartTokenizer
from KGBART_model.modeling_kgbart import KGBartForConditionalGeneration
from KGBART_model.optimization import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist
from prefetch_generator import BackgroundGenerator
import re
import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    fn_sched_list = glob.glob(os.path.join(output_dir, "sched.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list]) & set(
        [int(Path(fn).stem.split('.')[-1]) for fn in fn_sched_list])
    if both_set:
        return max(both_set)
    else:
        return None


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir",
                        default="../../../dataset/final_data/commongen",
                        type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--src_file", default=None, type=str,
                        help="The input data file name.")
    parser.add_argument("--tgt_file", default=None, type=str,
                        help="The output data file name.")
    parser.add_argument("--bart_model", default="facebook/bart-large", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--output_dir",
                        default="../../../output/train_kgbart",
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir",
                        default="../../../log/train_kgbart",
                        type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--optim_recover_path",
                        default=None,
                        type=str,
                        help="The file of pretraining optimizer.")
    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=64,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=True,
                        type=bool,
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        type=bool,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False, type=bool,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=48,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=2,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for attention probabilities.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=6,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', default=True, type=bool,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',
                        help="Whether to use 32-bit float precision instead of 16-bit for embeddings")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp', default=True, type=bool,
                        help="Whether to use amp for fp16")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--new_pos_ids', action='store_true',
                        help="Use new position ids for LMs.")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--max_len_a', type=int, default=64,
                        help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_b', type=int, default=64,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--trunc_seg', default='',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', default=True, type=bool,
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.70, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument("--mask_prob_eos", default=0, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=30,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=5, type=int,
                        help="Number of workers for the data loader.")

    parser.add_argument('--mask_source_words', action='store_true',
                        help="Whether to mask source words for training")
    parser.add_argument('--skipgram_prb', type=float, default=0.0,
                        help='prob of ngram mask')
    parser.add_argument('--skipgram_size', type=int, default=1,
                        help='the max size of ngram mask')
    parser.add_argument('--mask_whole_word', action='store_true',
                        help="Whether masking a whole word.")
    parser.add_argument('--do_l2r_training', action='store_true',
                        help="Whether to do left to right training")
    parser.add_argument('--pretraining_KG', action='store_true',
                        help="Whether to pre-training KG. ")
    parser.add_argument('--max_position_embeddings', type=int, default=64,
                        help="max position embeddings")
    parser.add_argument('--relax_projection', action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--ffn_type', default=0, type=int,
                        help="0: default mlp; 1: W((Wx+b) elem_prod x);")
    parser.add_argument('--num_qkv', default=0, type=int,
                        help="Number of different <Q,K,V>.")
    parser.add_argument('--seg_emb', action='store_true',
                        help="Using segment embedding for self-attention.")
    parser.add_argument('--s2s_special_token', default=False, type=bool,
                        help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
    parser.add_argument('--s2s_add_segment', default=False, type=bool,
                        help="Additional segmental for the encoder of S2S.")
    parser.add_argument('--s2s_share_segment', default=False, type=bool,
                        help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--keep_last_epochs", default=5, type=int, help="Keep the last few epochs.")

    args = parser.parse_args()

    # assert Path(args.model_recover_path).exists(
    # ), "--model_recover_path doesn't exist"

    args.output_dir = args.output_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
    args.log_dir = args.log_dir.replace(
        '[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    tokenizer = BartTokenizer.from_pretrained(
        args.bart_model, do_lower_case=args.do_lower_case)
    # if args.max_position_embeddings:
    #     tokenizer.max_len = args.max_position_embeddings
    data_tokenizer = tokenizer  # WhitespaceTokenizer() if args.tokenized_input else
    if args.local_rank == 0:
        dist.barrier()

    bi_uni_pipeline = [
        seq2seq_loader.Preprocess4Seq2seq(list(tokenizer.encoder.keys()), tokenizer.convert_tokens_to_ids,
                                          args.max_seq_length, new_segment_ids=args.new_segment_ids,
                                          truncate_config={'max_len_a': args.max_len_a,
                                                           'max_len_b': args.max_len_b,
                                                           'trunc_seg': args.trunc_seg,
                                                           'always_truncate_tail': args.always_truncate_tail},
                                          mode="s2s", pretraining_KG=args.pretraining_KG, num_qkv=args.num_qkv,
                                          s2s_special_token=args.s2s_special_token,
                                          s2s_add_segment=args.s2s_add_segment,
                                          s2s_share_segment=args.s2s_share_segment,
                                          pos_shift=args.pos_shift)]
    file_oracle = None
    # entity_id = os.path.join(
    #     args.data_dir, args.tgt_file if args.tgt_file else 'CommonGen_KG/commongen_entity2id.txt')
    # entity_embedding_path = os.path.join(
    #     args.data_dir, args.tgt_file if args.tgt_file else 'CommonGen_KG/commongen_ent_embeddings')
    #
    # relation_id = os.path.join(
    #     args.data_dir, args.tgt_file if args.tgt_file else 'CommonGen_KG/commongen_relation2id.txt')
    # relation_embedding_path = os.path.join(
    #     args.data_dir, args.tgt_file if args.tgt_file else 'CommonGen_KG/commongen_rel_embeddings')

    # TODO PROJECT_DIR 不应存成全局变量，要放入args里面
    entity_id = os.path.join(
        PROJECT_DIR, args.tgt_file if args.tgt_file else 'CommonGen_KG/commongen_entity2id.txt')
    entity_embedding_path = os.path.join(
        PROJECT_DIR, args.tgt_file if args.tgt_file else 'CommonGen_KG/commongen_ent_embeddings')

    relation_id = os.path.join(
        PROJECT_DIR, args.tgt_file if args.tgt_file else 'CommonGen_KG/commongen_relation2id.txt')
    relation_embedding_path = os.path.join(
        PROJECT_DIR, args.tgt_file if args.tgt_file else 'CommonGen_KG/commongen_rel_embeddings')

    entity_embedding = np.array(pickle.load(open(entity_embedding_path, "rb")))
    entity_embedding = np.array(list(np.zeros((4, 1024))) + list(entity_embedding))
    relation_embedding = np.array(pickle.load(open(relation_embedding_path, "rb")))

    if args.do_train:
        print("Loading Train Dataset", args.data_dir)
        # bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys(
        # )), tokenizer.convert_tokens_to_ids, args.max_seq_length, new_segment_ids=args.new_segment_ids, truncate_config={'max_len_a': args.max_len_a, 'max_len_b': args.max_len_b, 'trunc_seg': args.trunc_seg, 'always_truncate_tail': args.always_truncate_tail}, mask_source_words=args.mask_source_words, skipgram_prb=args.skipgram_prb, skipgram_size=args.skipgram_size, mask_whole_word=args.mask_whole_word, mode="s2s", has_oracle=args.pretraining_KG, num_qkv=args.num_qkv, s2s_special_token=args.s2s_special_token, s2s_add_segment=args.s2s_add_segment, s2s_share_segment=args.s2s_share_segment, pos_shift=args.pos_shift)]
        # file_oracle = None
        if args.pretraining_KG:
            file_oracle = os.path.join(args.data_dir, 'commongen.train.oracle')
        fn_src = os.path.join(
            args.data_dir, args.src_file if args.src_file else 'commongen.train.src_new.txt')
        fn_tgt = os.path.join(
            args.data_dir, args.tgt_file if args.tgt_file else 'commongen.train.tgt.txt')
        fn_onehop = os.path.join(
            args.data_dir, args.tgt_file if args.tgt_file else 'commongen.train.onehop_5.txt')
        train_dataset = seq2seq_loader.Seq2SeqDataset(
            fn_src, fn_tgt, entity_id, relation_id, fn_onehop, args.train_batch_size, data_tokenizer,
            args.max_seq_length, bi_uni_pipeline=bi_uni_pipeline)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset, replacement=False)
            _batch_size = args.train_batch_size
        else:
            train_sampler = DistributedSampler(train_dataset)
            _batch_size = args.train_batch_size // dist.get_world_size()
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler,
                                                       num_workers=args.num_workers,
                                                       collate_fn=seq2seq_loader.batch_list_to_batch_tensors,
                                                       pin_memory=False)
    if args.do_eval:
        print("Loading Dev Dataset", args.data_dir)
        if args.pretraining_KG:
            file_oracle = os.path.join(args.data_dir, 'commongen.dev.oracle')
        fn_src = os.path.join(
            args.data_dir, args.src_file if args.src_file else 'commongen.dev.src_new.txt')
        fn_tgt = os.path.join(
            args.data_dir, args.tgt_file if args.tgt_file else 'commongen.dev.tgt.txt')
        fn_onehop = os.path.join(
            args.data_dir, args.tgt_file if args.tgt_file else 'commongen.dev.onehop_5.txt')
        dev_dataset = seq2seq_loader.Seq2SeqDataset(
            fn_src, fn_tgt, entity_id, relation_id, fn_onehop, args.eval_batch_size, data_tokenizer,
            args.max_seq_length, bi_uni_pipeline=bi_uni_pipeline)
        if args.local_rank == -1:
            dev_sampler = RandomSampler(dev_dataset, replacement=False)
            _batch_size = args.eval_batch_size
        else:
            dev_sampler = DistributedSampler(dev_dataset)
            _batch_size = args.eval_batch_size // dist.get_world_size()
        dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=_batch_size,
                                                     sampler=dev_sampler,
                                                     num_workers=args.num_workers,
                                                     collate_fn=seq2seq_loader.batch_list_to_batch_tensors,
                                                     pin_memory=False)

    # note: args.train_batch_size has been changed to (/= args.gradient_accumulation_steps)
    # t_total = int(math.ceil(len(train_dataset.ex_list) / args.train_batch_size)
    t_total = int(len(train_dataloader) * args.num_train_epochs /
                  args.gradient_accumulation_steps)

    # Prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    cls_num_labels = 2
    type_vocab_size = 6 + \
                      (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
    num_sentlvl_labels = 2 if args.pretraining_KG else 0
    relax_projection = 4 if args.relax_projection else 0
    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    if (recover_step is None) and (args.model_recover_path is None):
        # if _state_dict == {}, the parameters are randomly initialized
        # if _state_dict == None, the parameters are initialized with bert-init
        _state_dict = {} if args.from_scratch else None
        model = KGBartForConditionalGeneration.from_pretrained(args.bart_model, entity_weight=entity_embedding,
                                                             relation_weight=relation_embedding)
        global_step = 0
    else:
        if recover_step:
            logger.info("***** Recover model: %d *****", recover_step)
            model_recover = torch.load(os.path.join(
                args.output_dir, "model.{0}.bin".format(recover_step)), map_location='cpu')
            # recover_step == number of epochs
            global_step = math.floor(
                recover_step * t_total / args.num_train_epochs)
        elif args.model_recover_path:
            logger.info("***** Recover model: %s *****",
                        args.model_recover_path)
            model_recover = torch.load(
                args.model_recover_path, map_location='cpu')
            global_step = 0
        model = KGBartForConditionalGeneration.from_pretrained(args.bart_model, state_dict=model_recover,
                                                             entity_weight=entity_embedding,
                                                             relation_weight=relation_embedding)
    if args.local_rank == 0:
        dist.barrier()

    model.to(device)
    if args.local_rank != -1:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("DistributedDataParallel")
        model = DDP(model, device_ids=[
            args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        # model = torch.nn.DataParallel(model)
        model = DataParallelImbalance(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)), map_location='cpu')
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)
        schedule_recover = torch.load(os.path.join(
            args.output_dir, "sched.{0}.bin".format(recover_step)), map_location='cpu')
        scheduler.load_state_dict(schedule_recover)

        if args.loss_scale == 0:
            logger.info("***** Recover optimizer: dynamic_loss_scale *****")
            optimizer.dynamic_loss_scale = True

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    best_dev_loss = 1000

    output_eval_file = os.path.join(args.log_dir, "eval_results.txt")
    writer = open(output_eval_file, "w")

    def checkpoint_paths(path, pattern=r"model(\d+)\.pt"):
        """Retrieves all checkpoints found in `path` directory.

        Checkpoints are identified by matching filename to the specified pattern. If
        the pattern contains groups, the result will be sorted by the first group in
        descending order.
        """
        pt_regexp = re.compile(pattern)
        files = os.listdir(path)

        entries = []
        for i, f in enumerate(files):
            m = pt_regexp.fullmatch(f)
            if m is not None:
                idx = float(m.group(1)) if len(m.groups()) > 0 else int(f.split(".")[1])
                entries.append((idx, m.group(0)))
        return [os.path.join(path, x[1]) for x in sorted(entries, reverse=True)]

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)

        if recover_step:
            start_epoch = recover_step + 1
        else:
            start_epoch = 1
        for i_epoch in trange(start_epoch, int(args.num_train_epochs) + 1, desc="Epoch",
                              disable=args.local_rank not in (-1, 0)):
            model.train()
            if args.local_rank != -1:
                train_sampler.set_epoch(i_epoch)
            # iter_bar = tqdm(BackgroundGenerator(train_dataloader), desc='Iter (loss=X.XXX)',
            #                 disable=args.local_rank not in (-1, 0))
            for step, batch in enumerate(tqdm(train_dataloader, desc="Training", position=0, leave=True)):
                batch = [
                    t.to(device) if t is not None else None for t in batch]
                if args.pretraining_KG:
                    input_ids, segment_ids, input_entity_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx, oracle_pos, oracle_weights, oracle_labels = batch
                else:
                    input_ids, input_entity_ids, subword_mask, word_mask, word_subword, decoder_input_ids, decoder_attention_mask, concept_entity_expand, concept_relation_expand, labels = batch
                    oracle_pos, oracle_weights, oracle_labels = None, None, None
                loss_output = model(input_ids, input_entity_ids=input_entity_ids, attention_mask=subword_mask,
                                    word_mask=word_mask, word_subword=word_subword,
                                    decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask, labels=labels,
                                    concept_entity_expand=concept_entity_expand,
                                    concept_relation_expand=concept_relation_expand,
                                    label_smoothing=False)

                masked_lm_loss = loss_output.loss
                if n_gpu > 1:  # mean() to average on multi-gpu.
                    # loss = loss.mean()
                    masked_lm_loss = masked_lm_loss.mean()
                loss = masked_lm_loss

                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                # iter_bar.set_description('Iter %d (loss=%5.3f)' % (i_epoch, loss.item()))

                if step % 1000 == 0:
                    print('Iter %d  (Gen_loss=%5.3f)' % (i_epoch, loss.item()))

                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

            if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                model.eval()
                cur_dev_loss = []
                with torch.no_grad():
                    for step, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating", position=0, leave=True)):
                        batch = [
                            t.to(device) if t is not None else None for t in batch]
                        if args.pretraining_KG:
                            input_ids, segment_ids, input_entity_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx, oracle_pos, oracle_weights, oracle_labels = batch
                        else:
                            input_ids, input_entity_ids, subword_mask, word_mask, word_subword, decoder_input_ids, decoder_attention_mask, concept_entity_expand, concept_relation_expand, labels = batch

                        loss_output = model(input_ids, input_entity_ids=input_entity_ids, attention_mask=subword_mask,
                                            word_mask=word_mask, word_subword=word_subword,
                                            decoder_input_ids=decoder_input_ids,
                                            decoder_attention_mask=decoder_attention_mask, labels=labels,
                                            concept_entity_expand=concept_entity_expand,
                                            concept_relation_expand=concept_relation_expand,
                                            label_smoothing=False)

                        masked_lm_loss = loss_output.loss

                        if n_gpu > 1:  # mean() to average on multi-gpu.
                            # loss = loss.mean()
                            masked_lm_loss = masked_lm_loss.mean()
                            # next_sentence_loss = next_sentence_loss.mean()
                        loss = masked_lm_loss
                        cur_dev_loss.append(float(loss.item()))
                        # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                    dev_loss = sum(cur_dev_loss) / float(len(cur_dev_loss))
                    print("the epoch {} DEV loss is {}".format(i_epoch, dev_loss))
                    if best_dev_loss > dev_loss:
                        best_dev_loss = dev_loss
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Only save the model it-self
                        os.makedirs(args.output_dir + "/best_model", exist_ok=True)
                        output_model_file = os.path.join(
                            args.output_dir, "best_model/model.best.bin")
                        # output_optim_file = os.path.join(
                        #     args.output_dir, "best_model/optim.best.bin")
                        # output_schedule_file = os.path.join(
                        #     args.output_dir, "best_model/sched.best.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        # torch.save(optimizer.state_dict(), output_optim_file)
                        # torch.save(scheduler.state_dict(), output_schedule_file)

                    logger.info(
                        "** ** * Saving fine-tuned model and optimizer ** ** * ")
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(
                        args.output_dir, "model.{0}.bin".format(i_epoch))
                    output_optim_file = os.path.join(
                        args.output_dir, "optim.{0}.bin".format(i_epoch))
                    output_schedule_file = os.path.join(
                        args.output_dir, "sched.{0}.bin".format(i_epoch))
                    torch.save(model_to_save.state_dict(), output_model_file)
                    torch.save(optimizer.state_dict(), output_optim_file)
                    torch.save(scheduler.state_dict(), output_schedule_file)

                    writer.write("epoch " + str(i_epoch) + "\n")
                    writer.write("the current eval accuracy is: " + str(dev_loss) + "\n")

                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

                    if args.keep_last_epochs > 0:
                        # remove old epoch checkpoints; checkpoints are sorted in descending order
                        checkpoints = checkpoint_paths(args.output_dir, pattern=r"model.\d+.bin")
                        for old_chk in checkpoints[args.keep_last_epochs:]:
                            if os.path.lexists(old_chk):
                                os.remove(old_chk)

                        checkpoints = checkpoint_paths(args.output_dir, pattern=r"optim.\d+.bin")
                        for old_chk in checkpoints[args.keep_last_epochs:]:
                            if os.path.lexists(old_chk):
                                os.remove(old_chk)

                        checkpoints = checkpoint_paths(args.output_dir, pattern=r"sched.\d+.bin")
                        for old_chk in checkpoints[args.keep_last_epochs:]:
                            if os.path.lexists(old_chk):
                                os.remove(old_chk)
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
