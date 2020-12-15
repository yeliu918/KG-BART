# KG-BART
KG-BART: Knowledge Graph-Augmented BART for Generative Commonsense Reasoning

## 0. Introduction
This is the official code base for the models in our paper on generative commonsense reasoning:

Ye Liu, Yao Wan, Lifang He, Hao Peng, Philip S. Yu. KG-BART: Knowledge Graph-Augmented BART for Generative Commonsense Reasoning. In AAAI 2021. (https://arxiv.org/abs/2009.12677) This code base is designed to reproduce our experiments in our paper, and can be used with other datasets. For any technical details, please refer to our paper.
When using our code or the methods, please cite our paper: [reference](https://arxiv.org/abs/2009.12677).

## 1. Knowledge Graph Grounding
Option 1:
Download our processed data:
1. [CommonGen](https://drive.google.com/drive/folders/1sOuSY4ZeXsf1vYbPumiQxg2Pr1CECJNk?usp=sharing)
2. [Entity Relation Embedding](https://drive.google.com/drive/folders/13h0PM_WvdsEh2FGc5l0iaxf7bWl_YVUe?usp=sharing)
3. [Concept Graph](https://drive.google.com/drive/folders/1i0UYYbUYNN4fmVKD5WgpImtcsRGWATnE?usp=sharing)

Option 2: 
Following our guidness and process from the original data:

Download the [Conceptnet](https://github.com/commonsense/conceptnet5/wiki/Downloads)

Download the [CommonGen Dataset](https://inklab.usc.edu/CommonGen/)

### 1.1 Preparing data 
```
./get_data.sh
```
### 1.2 Training Entity and Relation Embedding using TransE
```
./get_data.sh
```
## 2. Graph-Based Encoder-Decoder Modeling

### 2.0 Pre-training 
The following command shows how to pretrain our KG-BART model with the Conceptnet dataset created by ./get_data.sh as described above:
```
python pretrain_kgbart.py --data_dir  ../../dataset/commongen_data/commongen --output_dir ../../output/BART_KG  
  --log_dir ../../log/BART_KG --fp16 True --max_seq_length 32 --max_position_embeddings 64  --max_len_a 32 
  --max_len_b 64 --max_pred 64 --train_batch_size 128 --train_batch_size 24 --gradient_accumulation_steps 6
  --learning_rate 0.00001 --warmup_proportion 0.1 --label_smoothing 0.1 --num_train_epochs 10
```

### 2.1 Train and evaluate CommonGen
The following command shows how to fine-tune our KG-BART model with the CommonGen dataset created by ./get_data.sh as described above:
```
python run_seq2seq.py --data_dir  ../../dataset/commongen_data/commongen --output_dir ../../output/BART_KG  
  --log_dir ../../log/BART_KG --fp16 True --max_seq_length 32 --max_position_embeddings 64  --max_len_a 32 
  --max_len_b 64 --max_pred 64 --train_batch_size 128 --train_batch_size 24 --gradient_accumulation_steps 6 
  --learning_rate 0.00001  --warmup_proportion 0.1 --label_smoothing 0.1 --num_train_epochs 10
```
### 2.1 Test CommonGen
Finally, we can evalaute the models on the test set, by running the following command:
```
python decode_seq2seq.py --model_recover_path ../../output/BART_KG_new/model.best.bin 
  --input_file ../../dataset/commongen_data/commongen/commongen.test.src_alpha.txt 
  --output_dir ../../output/BART_v2/Gen --output_file model.best --split train --beam_size 5
```
## 3. Evaluation
Download the Evaluation Package and follow the README:
[Evaluation](https://github.com/INK-USC/CommonGen/tree/master/evaluation)

## Dependencies
* Python 3 (tested on python 3.6)
* PyTorch (with GPU and CUDA enabled installation (though the code is runnable on CPU, it would be way too slow))
* tqdm

