import json
from typing import List
from tqdm import tqdm
import glob
import os
from collections import defaultdict
from extract_entity import classify_all_entity


def entities_to_sentence(entities):
    fq = defaultdict(int)
    for ent in entities:
        fq[ent] += 1
    return ', '.join([f'{v} {k}' for k, v in fq.items()])


fols = list(glob.glob(
    '/home/khangln/JAIST_DRIVE/WORK/IconQA/data/iconqa_data/iconqa/*/*/*'))
data = []
for t in tqdm(fols):
    pid = os.path.basename(t)
    data_path = os.path.join(t, 'data.json')
    with open(data_path) as f:
        d = json.load(f)
        # Generate context sentence
        img_path = os.path.join(t, 'image.png')
        img_ents = classify_all_entity(img_path)
        context = entities_to_sentence(img_ents)
        data.append({
            **d,
            'pid': pid,
            'caption': context
        })


with open('all_captions.json', 'w') as f:
    f.write('\n'.join([json.dumps(x) for x in data]))
