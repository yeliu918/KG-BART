import csv
from tqdm import tqdm
import pandas as pd
from itertools import permutations
import argparse
from multiprocessing import Pool
import enchant
import random
import os
d = enchant.Dict("en_US")
import spacy
nlp = spacy.load('en_core_web_sm')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", default="Dataset", type =str,
                    help="The dataset dictionary")
parser.add_argument("--save_dataset_dir", default="Dataset", type=str,
                    help="The save processed data to saving dataset dictionary")
parser.add_argument("--org_conceptnet",
                    default="conceptnet-assertions-5.7.0.csv",
                    type=str,
                    help="The input conceptnet dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--save_conceptnet",
                    default="conceptnet.csv",
                    type=str,
                    help="The output conceptnet data dir.")

args = parser.parse_args()
org_conceptnet = os.path.join(args.dataset_dir, args.org_conceptnet)
conceptnet = os.path.join(args.dataset_dir, args.save_conceptnet)
commongend = os.path.join(args.dataset_dir, args.commongend)
os.makedirs(args.save_dataset_dir, exist_ok=True)

## convert to conceptnet.csv
with open(org_conceptnet) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    relation_list = []
    start_list = []
    end_list = []
    for row in tqdm(csv_reader):
        row_str = " ".join(row)
        relation = row_str.split('\t')[1].replace('/r/', '')
        relation_list.append(relation)
        start = row_str.split('\t')[2].split('/')[3]
        start_list.append(start)
        end = row_str.split('\t')[3].split('/')[-1]
        end_list.append(end)
    dataframe = pd.DataFrame({'relation': relation_list, 'start_node': start_list, 'end_node': end_list})
    dataframe.to_csv(conceptnet, index=False, sep=',')

## only set the commongen related entity to OpenKE
entity_time = []

with open(conceptnet) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    conceptnet_dict = {}
    for i, row in tqdm(enumerate(csv_reader)):
        if i == 0: continue
        if row[1] not in conceptnet_dict.keys():
            if row[0] not in ['ExternalURL','dbpedia/language']:
                conceptnet_dict[row[1]] = {row[2]:[row[0]]}
        else:
            if row[2] not in conceptnet_dict[row[1]].keys():
                if row[0] not in ['ExternalURL', 'dbpedia/language']:
                    conceptnet_dict[row[1]].update({row[2]:[row[0]]})
            else:
                if row[0] not in ['ExternalURL', 'dbpedia/language']:
                    conceptnet_dict[row[1]][row[2]].append(row[0])
print(len(conceptnet_dict.keys()))

comgen_all_entity = set()
for i, concept_dict in enumerate([commongend.replace('dev', 'train'), commongend, commongend.replace('dev', 'test')]):
    with open(concept_dict) as f_concept:
        concept_list = f_concept.readlines()

    relation = set()
    all_concept_new = []
    for concept in tqdm(concept_list):
        all_concept = concept.split()
        pairs = permutations(all_concept, 2)
        new_concept_order = []
        for pa in pairs:
            two_src = pa[0]
            two_tgt = pa[1]
            try:
                rel = conceptnet_dict[two_src][two_tgt]
                if two_src not in new_concept_order and two_tgt not in new_concept_order:
                    new_concept_order.append(two_src)
                    new_concept_order.append(two_tgt)
                elif two_src not in new_concept_order:
                    new_concept_order.insert(two_src, new_concept_order.index(two_tgt))
                elif two_tgt not in new_concept_order:
                    new_concept_order.insert(two_tgt, new_concept_order.index(two_src))
            except:
                continue
        for concept in all_concept:
            if concept not in new_concept_order:
                new_concept_order.append(concept)
        all_concept_new.append(new_concept_order)
    with open(concept_dict.replace("src_alpha", "src_new"),'w') as c_f:
        for concept_list in all_concept_new:
            c_f.write(" ".join(concept_list)+"\n")