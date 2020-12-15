# KG-BART
KG-BART: Knowledge Graph-Augmented BART for Generative Commonsense Reasoning

## 0. Introduction
This is the official code base for the models in our paper on generative commonsense reasoning:

Ye Liu, Yao Wan, Lifang He, Hao Peng, Philip S. Yu. KG-BART: Knowledge Graph-Augmented BART for Generative Commonsense Reasoning. In AAAI 2021. (https://arxiv.org/abs/2009.12677) This code base is designed to reproduce our experiments in our paper, and can be used with other datasets. For any technical details, please refer to our paper.
When using our code or the methods, please cite our paper: [reference](https://arxiv.org/abs/2009.12677).

## 1. Knowledge Graph Grounding
Download the [Conceptnet](https://github.com/commonsense/conceptnet5/wiki/Downloads)

Download the [CommonGen](https://drive.google.com/drive/folders/1sOuSY4ZeXsf1vYbPumiQxg2Pr1CECJNk?usp=sharing)

### 1.1 Preparing data for TransE
```
./get_data.sh
```
### 1.2 Training Entity and Relation Embedding using TransE
```
./get_data.sh
```
## 2. Graph-Based Encoder-Decoder Modeling

### 2.1 Train and evaluate CommonGen
```
python run_seq2seq.py --data_dir  ../../dataset/commongen_data/commongen --output_dir ../../output/BART_KG  
  --log_dir ../../log/BART_KG --fp16  --max_seq_length 32 --max_position_embeddings 64  --max_len_a 32 --max_len_b 64  
  --max_pred 64 --train_batch_size 128 --train_batch_size 24 --gradient_accumulation_steps 6 --learning_rate 0.00001
  --warmup_proportion 0.1 --label_smoothing 0.1 --num_train_epochs 10
```
### 2.1 Test CommonGen
```
python decode_seq2seq.py --model_recover_path ../../output/BART_KG_new/model.best.bin 
  --input_file ../../dataset/commongen_data/commongen/commongen.test.src_alpha.txt 
  --output_dir ../../output/BART_v2/Gen --output_file model.best --split train --beam_size 5
```
