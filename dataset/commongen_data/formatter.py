import json 
import sys
import os
import random 

random.seed(42) 

train_onehop_path = "/home/yeliu/Project/CommonGen/BART/Dataset/conceptnet_train.txt"
dev_onehop_path = "/home/yeliu/Project/CommonGen/BART/Dataset/conceptnet_dev.txt"

filenames = ["commongen.train.jsonl", "commongen.dev.jsonl"]
onehop_paths = [train_onehop_path, dev_onehop_path]
dirpath = "commongen"
if not os.path.exists(dirpath):
    os.mkdir(dirpath)
wrong_list = ['coffere', 'freestande', 'tufte', 'bobsle', 'sterle', 'preppe', 'woode', 'bricke', 'concer',
              'christma', 'purebre', 'passanger', 'trelli', 'boee', 'redheade', 'rhinocero']

correct_list = ['coffer', 'freestanding', 'tuft', 'bobsled', 'sterling', 'prep', 'wood', 'brick', 'concert',
                'christmas', 'purebred', 'passenger', 'trellis', 'boeing', 'redhead', 'rhinoceros']

test_onehop_path = "/home/yeliu/Project/CommonGen/BART/Dataset/conceptnet_test.txt"

assert len(wrong_list) == len(correct_list)
def split_file(filename, onehop_path):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            data.append(json.loads(line))

    with open(onehop_path, 'r') as one_f:
        onehop = one_f.readlines()
    assert len(data) == len(onehop)

    alpha_order_cs_str = []
    ori_cs_str = [] 
    idx_list = []
    idx = 0
    scenes_list = []
    reasons_list = []
    for item, item2 in zip(data, onehop):
        rand_order = []

        for word in item['concept_set'].split('#'):
            if word in wrong_list:
                correct_word = correct_list[wrong_list.index(word)]
                rand_order.append(correct_word)
            else:
                rand_order.append(word)

        # random.shuffle(rand_order)
        alpha_order = ' '.join(rand_order)

        scenes = item['scene'] 
        scenes_list += scenes
        ori_cs_str += [item['concept_set']] * len(scenes)
        alpha_order_cs_str += [alpha_order] * len(scenes) 
        idx_list += [str(idx)] * len(scenes)
        idx += 1
    prefix = filename.replace(".jsonl", "")
    with open(dirpath + "/%s.src_alpha.txt"%prefix, 'w', encoding="utf8") as f:
        f.write("\n".join(alpha_order_cs_str)) 
    with open(dirpath + "/%s.cs_str.txt"%prefix, 'w', encoding="utf8") as f:
        f.write("\n".join(ori_cs_str))
    with open(dirpath + "/%s.tgt.txt"%prefix, 'w', encoding="utf8") as f:
        f.write("\n".join(scenes_list))
    with open(dirpath + "/%s.index.txt"%prefix, 'w', encoding="utf8") as f:
        f.write("\n".join(idx_list)) 

for filename, onehop_path in zip(filenames, onehop_paths):
    split_file(filename, onehop_path)

def split_file_test(filename):
    data = []
    with open(filename) as f:
        for line in f.readlines():
            data.append(json.loads(line))
    alpha_order_cs_str = []
    ori_cs_str = []
    for item in data:
        # rand_order = [word for word in item['concept_set'].split('#')]
        rand_order = []
        for word in item['concept_set'].split('#'):
            if word in wrong_list:
                correct_word = correct_list[wrong_list.index(word)]
                rand_order.append(correct_word)
            else:
                rand_order.append(word)
        alpha_order = ' '.join(rand_order)
        ori_cs_str += [item['concept_set']]
        alpha_order_cs_str += [alpha_order]
    prefix = filename.replace(".jsonl", "")
    with open(dirpath + "/%s.src_alpha.txt"%prefix, 'w', encoding="utf8") as f:
        f.write("\n".join(alpha_order_cs_str))
    with open(dirpath + "/%s.cs_str.txt"%prefix, 'w', encoding="utf8") as f:
        f.write("\n".join(ori_cs_str))


split_file_test("commongen.test.jsonl")
