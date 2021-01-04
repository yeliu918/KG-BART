import csv
from tqdm import tqdm
import pandas as pd
from itertools import permutations
from multiprocessing import Pool
import enchant
import random
import json
import pickle
d = enchant.Dict("en_US")
import spacy

nlp = spacy.load('en_core_web_sm')
import argparse
import os

parser = argparse.ArgumentParser()

# Required parameters
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


args = parser.parse_args()

conceptnet = os.path.join(args.dataset_dir, args.save_conceptnet)
commongend = os.path.join(args.dataset_dir, args.commongend)
os.makedirs(args.save_dataset_dir, exist_ok = True)
os.makedirs(args.OpenKE_dir, exist_ok = True)

if 1:
    entity_time = []
    all_entity = []
    for file_name in [commongend, commongend.replace("dev", 'train'), commongend.replace("dev", 'test')]:
        train_entity = []
        dev_entity = []
        test_entity = []
        with open(file_name) as f_concept:
            concept_list = f_concept.readlines()
            for clist in concept_list:
                entitys = clist.split()
                for entity in entitys:
                    if entity not in all_entity:
                        all_entity.append(entity)
                    if "train" in file_name:
                        if entity not in train_entity:
                            train_entity.append(entity)
                    if "dev" in file_name:
                        if entity not in dev_entity:
                            dev_entity.append(entity)
                    if "test" in file_name:
                        if entity not in test_entity:
                            test_entity.append(entity)


    all_entity_file = os.path.join(args.save_dataset_dir,"all_entity.pickle")
    train_entity = os.path.join(args.save_dataset_dir,"train_entity.pickle")
    dev_entity = os.path.join(args.save_dataset_dir,"dev_entity.pickle")
    test_entity = os.path.join(args.save_dataset_dir,"test_entity.pickle")
    pickle.dump(all_entity, open(all_entity_file, "wb"))
    pickle.dump(train_entity, open(train_entity, "wb"))
    pickle.dump(dev_entity, open(dev_entity, "wb"))
    pickle.dump(test_entity, open(test_entity, "wb"))

    train_entity = os.path.join(args.save_dataset_dir,"train_entity.pickle")
    dev_entity = os.path.join(args.save_dataset_dir,"dev_entity.pickle")
    test_entity = os.path.join(args.save_dataset_dir,"test_entity.pickle")
    train_entity = pickle.load(open(train_entity, "rb"))
    dev_entity = pickle.load(open(dev_entity, "rb"))
    test_entity = pickle.load(open(test_entity, "rb"))

    all_entity = os.path.join(args.save_dataset_dir,"all_entity.pickle")
    comgen_entity= pickle.load(open(all_entity, "rb"))

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
            # if row[2] not in conceptnet_dict.keys():
            #     if row[0] not in ['ExternalURL']:
            #         conceptnet_dict[row[2]] = {row[1]:["NAN"+row[0]]}
            # else:
            #     if row[1] not in conceptnet_dict[row[2]].keys():
            #         if row[0] not in ['ExternalURL']:
            #             conceptnet_dict[row[2]].update({row[1]:["NAN"+row[0]]})
            #     else:
            #         if row[0] not in ['ExternalURL']:
            #             conceptnet_dict[row[2]][row[1]].append("NAN"+row[0])
    print(len(conceptnet_dict.keys()))
    comgen_all_entity = set()
    for i, concept_dict in enumerate([commongend.replace('dev', 'train'), commongend, commongend.replace('dev', 'test')]):
        with open(concept_dict) as f_concept:
                concept_list = f_concept.readlines()
                concept_set = []
                for concept in concept_list:
                    if concept not in concept_set:
                        concept_set.append(concept)
        relation = set()
        for concept in tqdm(concept_set):
            all_concept = concept.split()
            pairs = permutations(all_concept, 2)
            for pa in pairs:
                two_src = pa[0]
                two_tgt = pa[1]
                try:
                    for rel1 in conceptnet_dict[two_src][two_tgt]:
                        relation.add(",".join([two_src.replace(" ","_"), two_tgt.replace(" ","_"), rel1]))
                        comgen_all_entity.add(two_src.replace(" ","_"))
                        comgen_all_entity.add(two_tgt.replace(" ","_"))
                except:
                    continue
                try:
                    for tgt1 in conceptnet_dict[two_src].keys():
                        relation_1 = conceptnet_dict[two_src][tgt1]
                        if two_tgt in conceptnet_dict[tgt1].keys():
                            relation_2 = conceptnet_dict[tgt1][two_tgt]
                            for rel1 in relation_1:
                                relation.add(",".join([two_src.replace(" ","_"), tgt1.replace(" ","_"), rel1]))
                                comgen_all_entity.add(two_src.replace(" ", "_"))
                                comgen_all_entity.add(tgt1.replace(" ", "_"))
                            for rel2 in relation_2:
                                relation.add(",".join([tgt1.replace(" ","_"), two_tgt.replace(" ","_"), rel2]))
                                comgen_all_entity.add(tgt1.replace(" ", "_"))
                                comgen_all_entity.add(two_tgt.replace(" ", "_"))
                except:
                    continue

                try:
                    for tgt1 in conceptnet_dict[two_src].keys():
                        relation_1 = conceptnet_dict[two_src][tgt1]
                        for tgt2 in conceptnet_dict[tgt1].keys():
                            relation_2 = conceptnet_dict[tgt1][tgt2]
                            if pa[1] in conceptnet_dict[tgt2].keys():
                                relation_3 = conceptnet_dict[tgt2][pa[1]]
                                for rel1 in relation_1:
                                    relation.add(",".join([pa[0].replace(" ","_"), tgt1.replace(" ","_"), rel1]))
                                    comgen_all_entity.add(pa[0].replace(" ", "_"))
                                    comgen_all_entity.add(tgt1.replace(" ", "_"))
                                for rel2 in relation_2:
                                    relation.add(",".join([tgt1.replace(" ","_"), tgt2.replace(" ","_"), rel2]))
                                    comgen_all_entity.add(tgt1.replace(" ", "_"))
                                    comgen_all_entity.add(tgt2.replace(" ", "_"))
                                for rel3 in relation_3:
                                    relation.add(",".join([tgt2.replace(" ","_"), pa[1].replace(" ","_"), rel3]))
                                    comgen_all_entity.add(tgt2.replace(" ", "_"))
                                    comgen_all_entity.add(pa[1].replace(" ", "_"))
                                # relation.append(" ".join([pa[0], relation_1, tgt1, relation_2, tgt2, relation_3, pa[1]]))
                        else:
                            continue
                except:
                    continue

        if i == 0:
            train_relation = relation
        elif i == 1:
            dev_relation = relation
        elif i == 2:
            test_relation = relation

    all_relation = list(train_relation) + list(dev_relation) + list(test_relation)

    uniq_relation = list(set(all_relation))
    print("the total triple ", len(uniq_relation))

    train_relation = list(set(train_relation))
    print("the train triple ", len(train_relation))
    dev_relation = list(set(dev_relation))
    print("the dev triple ", len(dev_relation))
    test_relation = list(set(test_relation))
    print("the test triple ", len(test_relation))

    oov_entity = []
    in_entity = []
    for entity in comgen_entity:
        if entity not in list(comgen_all_entity):
            oov_entity.append(entity)
        else:
            in_entity.append(entity)
    print("the oov_entity: ", len(oov_entity))

    relation = []
    for two_src in in_entity:
        try:
            three_hop = []
            for tgt1 in conceptnet_dict[two_src].keys():
                relation_1 = conceptnet_dict[two_src][tgt1]
                relation_type = random.choices(relation_1, k=1)
                for rel in relation_type:
                    three_hop.append(",".join(
                        [two_src.replace(" ", "_"), tgt1.replace(" ", "_"), rel.replace(" ", "_")]))
        except:
            continue
        if len(three_hop) > 3:
            uniq_relation = random.choices(three_hop, k=3)
        else:
            uniq_relation = three_hop
        for relat in uniq_relation:
            ent1, ent2, rel = relat.split(',')
            comgen_all_entity.add(ent1)
            comgen_all_entity.add(ent2)
            relation.append(relat)

    for entity in tqdm(oov_entity):
        three_hop = set()
        try:
            for tgt1 in conceptnet_dict[entity].keys():
                relation_1 = conceptnet_dict[entity][tgt1]
                relation_type = random.choices(relation_1, k = 1)
                for rel in relation_type:
                    three_hop.add(",".join([entity.replace(" ","_"), tgt1.replace(" ","_"), rel.replace(" ","_")]))
        except:
            continue

        three_hop = list(three_hop)
        if len(list(three_hop)) > 4:
            uniq_relation = random.choices(three_hop, k=4)
        else:
            uniq_relation = three_hop
        for relat in uniq_relation:
            ent1, ent2, rel = relat.split(',')
            comgen_all_entity.add(ent1)
            comgen_all_entity.add(ent2)
            relation.append(relat)

        try:
            three_hop = set()
            for tgt1 in conceptnet_dict[entity].keys():
                relation_1 = conceptnet_dict[entity][tgt1]
                relation_type1 = random.choices(relation_1, k=1)
                for tgt2 in conceptnet_dict[tgt1].keys():
                    relation_2 = conceptnet_dict[tgt1][tgt2]
                    relation_type2 = random.choices(relation_2, k=1)
                    three_hop.add(",".join([entity.replace(" ", "_"), relation_type1[0].replace(" ", "_"), tgt1.replace(" ", "_"), relation_type2[0].replace(" ", "_"), tgt2.replace(" ", "_")]))
                    continue
        except:
            continue

        three_hop = list(three_hop)
        if len(three_hop) > 4:
            uniq_relation = random.choices(three_hop, k=4)
        else:
            uniq_relation = three_hop

        for relat in uniq_relation:
            ent1,  rel1, ent2, rel2, ent3 = relat.split(',')
            comgen_all_entity.add(ent1)
            comgen_all_entity.add(ent2)
            comgen_all_entity.add(ent3)
            relation.append(",".join([ent1, ent2, rel1]))
            relation.append(",".join([ent2, ent3, rel2]))

    print("the relation of the oov ", len(relation))

    oov_entity = []
    for entity in comgen_entity:
        if entity not in list(comgen_all_entity):
            oov_entity.append(entity)

    print("the oov_entity: ", len(oov_entity))
    print("the oov entity: ", oov_entity)
    all_relation = all_relation + relation

    # three_hop = set()
    # for entity in tqdm(comgen_entity):
    #     try:
    #         for tgt1 in conceptnet_dict[entity].keys():
    #             relation_1 = conceptnet_dict[entity][tgt1]
    #             for relation_type in relation_1:
    #                 if relation_type == 'dbpedia/language'  or relation_type == 'ExternalURL' or relation_type == "EtymologicallyDerivedFrom": continue
    #                 three_hop.add(",".join([entity.replace(" ","_"), tgt1.replace(" ","_"), relation_type.replace(" ","_")]))
    #             # for tgt2 in conceptnet_dict[tgt1].keys():
    #             #     relation_2 = " ".join(conceptnet_dict[tgt1][tgt2])
    #             #     for relation_type in relation_2:
    #             #         three_hop.add(",".join([tgt1.replace(" ","_"), tgt2.replace(" ","_"), relation_type.replace(" ","_")]))
    #                 # for tgt3 in conceptnet_dict[tgt2].keys():
    #                 #     relation_3 = " ".join(conceptnet_dict[tgt2][tgt3])
    #                 #     three_hop.add(" ".join([tgt2, tgt3, relation_3]))
    #     except:
    #         continue

    two_hop_common = os.path.join(args.save_dataset_dir, 'two_hop_commonnet.pickle')
    uniq_relation = list(set(all_relation))
    print("the total triple ", len(uniq_relation))
    pickle.dump(uniq_relation, open(two_hop_common,"wb"))

    uniq_relation = pickle.load(open(two_hop_common,"rb"))
    comgen_triple = set()
    comgen_triple_dict = {}
    comgen_all_entity = set()
    comgen_relation = set()
    for rel in tqdm(uniq_relation):
        entity1, entity2, relation = rel.split(',')
        new_entity1, new_entity2, new_relation = rel.split(',')
        if (not new_entity1.replace("_",'').encode( 'UTF-8' ).isalpha() or not new_entity2.replace("_",'').encode( 'UTF-8' ).isalpha()) and (new_entity1 not in comgen_entity and new_entity2 not in comgen_entity): continue
        comgen_all_entity.add(entity1)
        comgen_all_entity.add(entity2)
        comgen_relation.add(relation.replace("NAN",""))
        if "NAN" in relation:
            triple = ",".join([entity2, entity1, relation.replace("NAN","")])
            if " ".join([entity2, entity1]) not in comgen_triple_dict:
                comgen_triple_dict[" ".join([entity2, entity1])] = [relation.replace("NAN","")]
            else:
                comgen_triple_dict[" ".join([entity2, entity1])].append(relation.replace("NAN",""))

        else:
            triple = rel
            if " ".join([entity1, entity2]) not in comgen_triple_dict:
                comgen_triple_dict[" ".join([entity1, entity2])] = [relation]
            else:
                comgen_triple_dict[" ".join([entity1, entity2])].append(relation.replace("NAN",""))

        comgen_triple.add(triple)

    train_triple = list(comgen_triple)

    dev_triple = set()
    for rel in dev_relation:
        entity1, entity2, relation = rel.split(',')
        new_entity1, new_entity2, new_relation = rel.split(',')
        if not new_entity1.replace("_", '').encode('UTF-8').isalpha() or not new_entity2.replace("_", '').encode(
            'UTF-8').isalpha(): continue
        if "NAN" in relation:
            triple = ",".join([entity2, entity1, relation.replace("NAN", "")])
        else:
            triple = rel
        dev_triple.add(triple)
    dev_triple = list(dev_triple)

    test_triple = set()
    for rel in test_relation:
        entity1, entity2, relation = rel.split(',')
        new_entity1, new_entity2, new_relation = rel.split(',')
        if not new_entity1.replace("_", '').encode('UTF-8').isalpha() or not new_entity2.replace("_", '').encode(
            'UTF-8').isalpha(): continue
        if "NAN" in relation:
            triple = ",".join([entity2, entity1, relation.replace("NAN", "")])
        else:
            triple = rel
        test_triple.add(triple)

    test_triple = list(test_triple)

    oov_entity = []
    for entity in comgen_entity:
        if entity not in list(comgen_all_entity):
            oov_entity.append(entity)

    print("the oov_entity: ", len(oov_entity))
    print("the oov entity: ", oov_entity)

    print("entity number ", len(list(comgen_all_entity)))
    print("relation number ", len(list(comgen_relation)))
    print(comgen_relation)
    uniq_triple = list(comgen_triple)
    print("the total triple ", len(uniq_triple))
    # dev_triple = []
    # test_triple = []
    # commongend = '../CommonGen/dataset/commongen_data/commongen/commongen.dev.src_alpha.txt'
    # for concept_dict in [commongend, commongend.replace('dev', 'test')]:
    #     with open(concept_dict) as f_concept:
    #         concept_list = f_concept.readlines()
    #         concept_set = []
    #         for concept in concept_list:
    #             if concept not in concept_set:
    #                 concept_set.append(concept)
    #         relation_list = []
    #         for concept in concept_set:
    #             all_concept = concept.split()
    #             pairs = permutations(all_concept, 2)
    #             for pair in pairs:
    #                 str_pair = " ".join(pair)
    #                 try:
    #                     if 'dev' in concept_dict:
    #                         for relation_type in comgen_triple_dict[str_pair]:
    #                             dev_triple.append(",".join([pair[0], pair[1], relation_type]))
    #                     elif "test" in concept_dict:
    #                         for relation_type in comgen_triple_dict[str_pair]:
    #                             test_triple.append(",".join([pair[0], pair[1], relation_type]))
    #                 except:
    #                     continue
    # test_triple = list(set(test_triple))
    # dev_triple = list(set(dev_triple))
    # train_triple = uniq_triple
    # dev_test_triple = list(set(dev_triple)|set(test_triple))
    train_tripe = uniq_relation

    print("the train triple ", len(train_tripe))
    print("the test triple ", len(test_triple))
    print("the dev triple ", len(dev_triple))

    train_file = os.path.join (args.OpenKE_dir, "train2id.txt")
    dev_file = os.path.join(args.OpenKE_dir, "valid2id.txt")
    test_file = os.path.join(args.OpenKE_dir, "test2id.txt")
    entity2id = os.path.join(args.OpenKE_dir, "entity2id.txt")
    relation2id = os.path.join(args.OpenKE_dir, "relation2id.txt")

    rel_dict = {}
    entity_dict = {}
    with open(relation2id, "w") as rel_f:
        rel_f.write(str(len(list(comgen_relation)))+"\n")
        for i, rel in enumerate(list(comgen_relation)):
            rel_dict[rel] = i
            rel_f.write(rel+" "+str(i)+"\n")

    with open(entity2id, "w") as entity_f:
        entity_f.write(str(len(list(comgen_all_entity)))+"\n")
        for i, ent in enumerate(list(comgen_all_entity)):
            entity_dict[ent] = i
            entity_f.write(ent +" "+ str(i) + "\n")

    with open(train_file, "w") as train_f:
        train_f.write(str(len(train_triple))+"\n")
        for tri in train_triple:
            entity1, entity2, relation = tri.split(",")
            new_list = " ".join([str(entity_dict[entity1]), str(entity_dict[entity2]), str(rel_dict[relation])])
            train_f.write(new_list+"\n")
    with open(dev_file, "w") as dev_f:
        dev_f.write(str(len(dev_triple))+"\n")
        for tri in dev_triple:
            entity1, entity2, relation = tri.split(",")
            new_list = " ".join([str(entity_dict[entity1]), str(entity_dict[entity2]), str(rel_dict[relation])])
            dev_f.write(new_list+"\n")
    with open(test_file, "w") as test_f:
        test_f.write(str(len(test_triple))+"\n")
        for tri in test_triple:
            entity1, entity2, relation = tri.split(",")
            new_list = " ".join([str(entity_dict[entity1]), str(entity_dict[entity2]), str(rel_dict[relation])])
            test_f.write(new_list+"\n")

if 0: ## generate the commongen concept graph
    wrong_list = ['coffere', 'freestande', 'tufte', 'bobsle', 'sterle', 'preppe', 'woode', 'bricke', 'concer',
                  'christma', 'purebre', 'passanger', 'trelli', 'boee', 'redheade', 'rhinocero']
    correct_list = ['coffer', 'freestanding', 'tuft', 'bobsled', 'sterling', 'prep', 'wood', 'brick', 'concert', 'christmas',
                    'purebred', 'passenger', 'trellis', 'boeing', 'redhead', 'rhinoceros']

    import json
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

    for i, concept_dict in enumerate([commongend.replace('dev', 'train'), commongend, commongend.replace('dev', 'test')]):
        data = []
        with open(concept_dict) as f:
            for line in f.readlines():
                data.append(json.loads(line))
        concept_list = []
        for item in data:
            rand_list = []
            for word in item['concept_set'].split('#'):
                if word in wrong_list:
                    correct_word = correct_list[wrong_list.index(word)]
                    rand_list.append(correct_word)
                else:
                    rand_list.append(word)
            concept_list.append(rand_list)

        all_order_set = []
        for concept in tqdm(concept_list):
            all_concept = concept
            pairs = permutations([all_concept], 2)
            relation = set()
            for pa in pairs:
                two_src = pa[0]
                two_tgt = pa[1]
                try:
                    for rel1 in conceptnet_dict[two_src][two_tgt]:
                        relation.add([two_src, two_tgt])
                except:
                    continue
            order_set = []
            for pair in relation:
                src, tgt = pair[0], pair[1]
                order_set.append([src, tgt])
