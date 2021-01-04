import csv
from tqdm import tqdm
import pandas as pd
from itertools import permutations
from multiprocessing import Pool
import enchant
import random
import pickle
import json
d = enchant.Dict("en_US")
import spacy
nlp = spacy.load('en_core_web_sm')
import argparse
import os
import torch
import torchtext

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="dataset", type=str,
                    help="The dataset dictionary")
parser.add_argument("--save_dataset_dir", default="save_dataset", type=str,
                    help="The save processed data to saving dataset dictionary")
parser.add_argument("--OpenKE_dir", default="OpenKE/benchmarks/CommonGen", type = str,
                    help="The dictionary to save the processed data for OpenKE")
parser.add_argument("--save_conceptnet",
                    default="conceptnet.csv",
                    type=str,
                    help="The output conceptnet data dir.")
parser.add_argument("--commongend",
                    default="commongen.dev.src_alpha.txt",
                    type=str,
                    help="The default dir of the commongen dataset.")
parser.add_argument("--commongenind",
                    default="commongen.dev.index.txt",
                    type=str,
                    help="The default dir of the commongen dataset.")

args = parser.parse_args()
org_conceptnet = os.path.join(args.dataset_dir, args.org_conceptnet)
conceptnet = os.path.join(args.dataset_dir, args.save_conceptnet)
commongend = os.path.join(args.dataset_dir, args.commongend)
indexf = os.path.join(args.dataset_dir, args.commongenind)
os.makedirs(args.save_dataset_dir, exist_ok=True)

if 1:
    with open(conceptnet) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        conceptnet_dict = {}
        for i, row in tqdm(enumerate(csv_reader)):
            if i == 0: continue
            if row[1] not in conceptnet_dict.keys():
                if row[0] not in ['ExternalURL', 'dbpedia/language']:
                    conceptnet_dict[row[1]] = {row[2]: [row[0]]}
            else:
                if row[2] not in conceptnet_dict[row[1]].keys():
                    if row[0] not in ['ExternalURL', 'dbpedia/language']:
                        conceptnet_dict[row[1]].update({row[2]: [row[0]]})
                else:
                    if row[0] not in ['ExternalURL', 'dbpedia/language']:
                        if row[0] not in conceptnet_dict[row[1]][row[2]]:
                            conceptnet_dict[row[1]][row[2]].append(row[0])

    all_entity = []
    for file_name in [commongend.replace("dev", 'train'), commongend, commongend.replace("dev", 'test')]: #
        with open(file_name.replace("src_alpha","index"), "r") as ind_f:
            index_line = ind_f.readlines()
            count = 0
            new_index_line = [l.replace("\n", "") for l in list(index_line)]
            singid = []
            flag = 0
            for i, index in enumerate(new_index_line):
                if index == str(flag):
                    singid.append(i)
                    flag += 1

        train_entity = []
        dev_entity = []
        test_entity = []
        uniq_list = []
        with open(file_name) as f_concept:
            concept_list = f_concept.readlines()
            concept_onehop = {}
            for ind in singid:
                uniq_list.append(concept_list[ind])

            print("the uniq_list:", len(uniq_list))
            for i, each_entity in tqdm(enumerate(uniq_list)):
                entitys = each_entity.split()
                onehop = {}
                for entity in entitys:
                    doc = nlp(entity)
                    entity_pos = [t.pos_ for t in doc][0]
                    try:
                        if entity_pos == "NOUN" or entity_pos == "PROPN" or entity_pos == "X" or entity_pos == "PROP":
                            for word in  conceptnet_dict[entity].keys():
                                if not word.replace("_",'').encode( 'UTF-8' ).isalpha(): continue
                                if not d.check(word.replace("_", '')) : continue
                                if '_' not in word:
                                    doc = nlp(word)
                                    word_pos = [t.pos_ for t in doc][0]
                                else:
                                    word_pos = "NAN"
                                    doc = nlp(word.replace("_",''))
                                    for t in doc:
                                        if t.pos_ == "ADJ":
                                            word_pos = "ADJ"
                                if word_pos == "ADJ":
                                    if entity not in onehop.keys():
                                        onehop[entity] = {word:conceptnet_dict[entity][word]}
                                    else:
                                        onehop[entity].update({word: conceptnet_dict[entity][word]})
                        elif entity_pos == "VERB" or entity_pos == "PROPN":
                            for word in conceptnet_dict[entity].keys():
                                if not word.replace("_", '').encode('UTF-8').isalpha(): continue
                                if not d.check(word.replace("_", '')): continue
                                if '_' not in word:
                                    doc = nlp(word)
                                    word_pos = [t.pos_ for t in doc][0]
                                else:
                                    word_pos = "NAN"
                                    doc = nlp(word.replace("_", ''))
                                    for t in doc:
                                        if t.pos_ == "ADV":
                                            word_pos = "ADV"
                                if word_pos == "ADV":
                                    if entity not in onehop.keys():
                                        onehop[entity] = {word:conceptnet_dict[entity][word]}
                                    else:
                                        onehop[entity].update({word:conceptnet_dict[entity][word]})
                        elif entity_pos == "ADJ":
                            for word in conceptnet_dict[entity].keys():
                                if not word.replace("_", '').encode('UTF-8').isalpha(): continue
                                if not d.check(word.replace("_", '')): continue
                                if '_' not in word:
                                    doc = nlp(word)
                                    word_pos = [t.pos_ for t in doc][0]
                                else:
                                    word_pos = "NAN"
                                    doc = nlp(word.replace("_", ''))
                                    for t in doc:
                                        if t.pos_ == "NOUN":
                                            word_pos = "NOUN"
                                if word_pos == "NOUN":
                                    if entity not in onehop.keys():
                                        onehop[entity] = {word: conceptnet_dict[entity][word]}
                                    else:
                                        onehop[entity].update({word: conceptnet_dict[entity][word]})
                        elif entity_pos == "ADV":
                            for word in conceptnet_dict[entity].keys():
                                if not word.replace("_", '').encode('UTF-8').isalpha(): continue
                                if not d.check(word.replace("_", '')): continue
                                if '_' not in word:
                                    doc = nlp(word)
                                    word_pos = [t.pos_ for t in doc][0]
                                else:
                                    word_pos = "NAN"
                                    doc = nlp(word.replace("_", ''))
                                    for t in doc:
                                        if t.pos_ == "VERB":
                                            word_pos = "VERB"
                                if word_pos == "VERB":
                                    if entity not in onehop.keys():
                                        onehop[entity] = {word: conceptnet_dict[entity][word]}
                                    else:
                                        onehop[entity].update({word: conceptnet_dict[entity][word]})
                        else:
                            onehop[entity] = {}
                    except:
                        continue

                concept_onehop[i]=onehop
                # if onehop != {}:
                #     print(onehop)
            print("the onehop_list:", len(concept_onehop))
            with  open(file_name.replace('src_alpha','onehop'),'w') as outfile:
                json.dump(concept_onehop, outfile)

if 0:
    clean_conceptnet_src = []
    clean_conceptnet_tgt =[]
    with open(conceptnet) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        conceptnet_dict = {}
        for i, row in tqdm(enumerate(csv_reader)):
            label = True
            for word in row:
                if not word.replace("_", '').encode('UTF-8').isalpha() or not d.check(word.replace("_", '')):
                    label = False
            if label == True:
                clean_conceptnet_src.append(" ".join(row[1:]))
                clean_conceptnet_tgt.append(row[0])
    print(str(len(clean_conceptnet_src))+" "+str(len(clean_conceptnet_tgt)))
    with open("../../../dataset/commongen_data/commongen/conceptnet_clean_src.txt",'w') as con_f:
        for concept in clean_conceptnet_src:
            con_f.write(concept)
    with open("../../../dataset/commongen_data/commongen/conceptnet_clean_tgt.txt", 'w') as con_f:
        for concept in clean_conceptnet_tgt:
            con_f.write(concept)
if 1:
    glove = torchtext.vocab.GloVe(name="6B", dim=50)
    commongend = commongend.replace('src_alpha','onehop')
    all_entity = []
    for file_name, index_file in zip([commongend.replace("dev", 'train')],[indexf.replace("dev","train")]): #commongend, commongend.replace("dev", 'train'),     indexf, indexf.replace("dev", "train"),
        with open(file_name,"r") as file:
            with open(index_file, "r") as ind_f:
                index_line = ind_f.readlines()
                count = 0
                new_index_line = [l.replace("\n","") for l in list(index_line)]
                sing2mul = dict((int(l), new_index_line.count(l)) for l in list(new_index_line))
                assert sum(sing2mul.values()) == len(index_line)
                concept_list = json.load(file)
                # assert concept_list[-2] == concept_list[-1]
                # concept_list = concept_list[0:-1]
                assert len(concept_list) == len(sing2mul)
                new_concept_list = []
                for index in tqdm(concept_list.keys()):
                    onehop = concept_list[index]
                    for hop in onehop.keys():
                        concept_node = list(onehop.keys())
                        filter_node = list(onehop[hop].keys())
                        rank_score = []
                        if len(filter_node) > 5:
                            for node in filter_node:
                                sum_score = 0
                                if len(node.split()) ==1:
                                    for each_con in concept_node:
                                        sum_score += torch.cosine_similarity(glove[node].unsqueeze(0), glove[each_con].unsqueeze(0)).item()
                                else:
                                    for each_con in concept_node:
                                        sub_score = 0
                                        for sub in node.split():
                                            sub_score += torch.cosine_similarity(glove[sub].unsqueeze(0),
                                                                             glove[each_con].unsqueeze(0)).item()
                                        sub_score /= len(node.split())
                                    sum_score += sub_score

                                rank_score.append(sum_score)
                            rank_score = sorted(range(len(rank_score)), key=lambda k: rank_score[k], reverse=True)[5:]
                        for i in rank_score:
                            onehop[hop].pop(filter_node[i])

                    for i in range(sing2mul[int(index)]):
                        new_concept_list.append(onehop)
                    # new_concept_list.append(onehop)

        assert len(new_concept_list) == len(index_line)
        print("the length of concept_list", len(new_concept_list))
        with open(file_name.replace('onehop','onehop_5'),'w') as outfile:
                json.dump(new_concept_list, outfile, indent=4)

if 1:
    all_entity_f = os.path.join(args.save_dataset_dir, "all_entity.pickle")
    comgen_entity= pickle.load(open(all_entity_f, "rb"))

    comgen_all_entity = set()
    train_one_hop =  commongend.replace('onehop','onehop_5')
    for file_name in [train_one_hop, train_one_hop.replace("dev", 'train'), train_one_hop.replace("dev", 'test')]:
        with open(file_name,"r") as file:
            concept_list = json.load(file)
            for onehop in tqdm(concept_list):
                for hop in onehop.keys():
                    for tgt in onehop[hop].keys():
                        comgen_all_entity.add(hop)
                        comgen_all_entity.add(tgt.replace(" ", "_"))

    all_entity = list(set(comgen_entity + list(comgen_all_entity)))
    print(len(all_entity))

    save_path = os.path.join(args.OpenKE_dir, "CommonGen_entity.pickle")
    pickle.dump(all_entity, open(save_path, "wb"))


