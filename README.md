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
Download the following data to dataset direction:

1. Download the [Conceptnet](https://github.com/commonsense/conceptnet5/wiki/Downloads)
2. Download the [CommonGen Dataset](https://inklab.usc.edu/CommonGen/), which require to fill the form

### 1.1 Preparing data  
```
python reorder_src.py --dataset_dir "dataset"  --save_dataset_dir "dataset/save_dataset" 
--org_conceptnet "conceptnet-assertions-5.7.0.csv"  --save_conceptnet "conceptnet.csv"
python conceptnet.py --dataset_dir "dataset"  --save_dataset_dir "dataset/save_dataset"   
--OpenKE_dir "../OpenKE/benchmarks/CommonGen"  --save_conceptnet "conceptnet.csv"
```
### 1.2 Training Entity and Relation Embedding using TransE
```
cd OpenKE
git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE
cd benchmarks/CommonGen
python n-n.py
python train_transe_CommonGen.py
cd ../KG_grounding
python entity_onehot.py --dataset_dir "dataset"  --save_dataset_dir "dataset/save_dataset"  
--OpenKE_dir "../OpenKE/benchmarks/CommonGen"  --save_conceptnet "conceptnet.csv"
```
## 2. Graph-Based Encoder-Decoder Modeling

### 2.0 Pre-training 
Option 2.0.1 The following command shows how to pretrain our KG-BART model with the Conceptnet dataset created by ./get_data.sh as described above:
```
python pretrain_kgbart.py --data_dir ../dataset/commongen_data/commongen --output_dir ../output/Pretraining_KG 
    --log_dir ../log/Pretraining_KG2 --pretraining_KG --train_pretraining_num 400000 --val_pretraining_num 80000 
    --fp16 True --max_seq_length 32 --max_position_embeddings 64 --max_len_a 32 --max_len_b 64 --max_pred 64 
    --train_batch_size 60 --eval_batch_size 48 --gradient_accumulation_steps 6 --learning_rate 0.00001 
    --warmup_proportion 0.1 --label_smoothing 0.1 --num_train_epochs 10
```
Option 2.0.2 Download the [Pre-training Weight](https://drive.google.com/drive/folders/18BHATG8ZtZiO6sLQemNWsLUdwwuqUabE?usp=sharing)

### 2.1 Train and evaluate CommonGen
The following command shows how to fine-tune our KG-BART model with the CommonGen dataset created by ./get_data.sh as described above:
```
python run_seq2seq.py --data_dir  ../dataset/commongen_data/commongen --output_dir ../output/KGBart
    --log_dir ../log/KGBart --model_recover_path ../output/Pretraining_KG/best_model/model.best.bin --fp16 True
    --max_seq_length 32 --max_position_embeddings 64 --max_len_a 32 --max_len_b 64 --max_pred 64
    --train_batch_size 60 --eval_batch_size 48 --gradient_accumulation_steps 6 --learning_rate 0.00001
    --warmup_proportion 0.1 --label_smoothing 0.1 --num_train_epochs 10
```
### 2.1 Test CommonGen
Finally, we can evalaute the models on the test set, by running the following command:
```
python decode_seq2seq.py --data_dir ../dataset/commongen_data/commongen --model_recover_path ../output/KGBart/best_model/model.best.bin
 --input_file ../dataset/commongen_data/commongen/commongen.dev.src_alpha.txt --output_dir ../output/KGBart/best_model/Gen
 --output_file model.best --split dev --beam_size 5 --forbid_duplicate_ngrams True
```
## 3. Evaluation
Download the Evaluation Package and follow the README:
[Evaluation](https://github.com/INK-USC/CommonGen/tree/master/evaluation)

## Dependencies
* Python 3 (tested on python 3.6)
* PyTorch (with GPU and CUDA enabled installation)
* amp 
* tqdm
## Acknowledgments
Our code is based on Huggingface Transformer. We thank the authors for their wonderful open-source efforts.