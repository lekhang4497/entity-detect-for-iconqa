import json
from typing import List
from tqdm import tqdm
import glob
import os
from collections import defaultdict
from extract_entity import classify_all_entity

SPLIT = 'test'
TYPE = 'choose_img'
ORIGIN_DATA_PATTERN = f'/home/khangln/JAIST_DRIVE/WORK/IconQA/data/iconqa_data/iconqa/{SPLIT}/{TYPE}/*'
PID2SKILL = '/home/khangln/JAIST_DRIVE/WORK/IconQA/data/iconqa_data/pid2skills.json'
# SKILL = 'algebra'

data_folders = sorted(glob.glob(ORIGIN_DATA_PATTERN))


def entities_to_sentence(entities):
    fq = defaultdict(int)
    for ent in entities:
        fq[ent] += 1
    return ', '.join([f'{v} {k}' for k, v in fq.items()])


def gen_data_from_choose_img(data_folders):
    data = []
    for t in tqdm(data_folders):
        pid = os.path.basename(t)
        data_path = os.path.join(t, 'data.json')
        with open(data_path) as f:
            d = json.load(f)
            question = d['question']
            # Generate context sentence
            img_path = os.path.join(t, 'image.png')
            img_ents = classify_all_entity(img_path)
            context = entities_to_sentence(img_ents)
            # Generate choice sentences
            choices: List[str] = []
            for choice_img in d['choices']:
                choice_path = os.path.join(t, choice_img)
                ents = classify_all_entity(choice_path)
                choice = entities_to_sentence(ents)
                choices.append(choice)
            # The number of choices must be 5
            while len(choices) < 5:
                choices.append('')
            label = d['answer']
            data.append({
                **{f'ending{i}': sent for i, sent in enumerate(choices)},
                'label': label,
                'sent1': context,
                'sent2': question,
                'pid': pid
            })
    return data


def gen_data_from_choose_txt(data_folders):
    data = []
    for t in tqdm(data_folders):
        pid = os.path.basename(t)
        data_path = os.path.join(t, 'data.json')
        with open(data_path) as f:
            d = json.load(f)
            question = d['question']
            # Generate context sentence
            img_path = os.path.join(t, 'image.png')
            img_ents = classify_all_entity(img_path)
            context = entities_to_sentence(img_ents)

            choices = d['choices']
            # The number of choices must be 5
            while len(choices) < 5:
                choices.append('')

            label = d['answer']
            data.append({
                **{f'ending{i}': sent for i, sent in enumerate(choices)},
                'label': label,
                'sent1': context,
                'sent2': question,
                'pid': pid
            })
    return data


def gen_data_from_fill_in_blank(data_folders):
    data = []
    for t in tqdm(data_folders):
        pid = os.path.basename(t)
        data_path = os.path.join(t, 'data.json')
        with open(data_path) as f:
            d = json.load(f)
            question = d['question']
            # Generate context sentence
            img_path = os.path.join(t, 'image.png')
            img_ents = classify_all_entity(img_path)
            context = entities_to_sentence(img_ents)
            label = d['answer']
            data.append({
                'pid': pid,
                'target': label,
                'source': question + " Context: " + context
            })
    return data


# --- For generate test set for specific skill ---
# with open(PID2SKILL) as f:
#     pid2skill = json.load(f)

# all_skills = set([x for v in pid2skill.values() for x in v])
# print('Skills:', all_skills)

# for SKILL in all_skills:
#     print(f'Generate test {TYPE} for skill: {SKILL}')
#     counting_fols = []
#     for fol in data_folders:
#         pid = os.path.basename(fol)
#         if SKILL in pid2skill[pid]:
#             counting_fols.append(fol)

#     if TYPE == 'choose_img':
#         gen_func = gen_data_from_choose_img
#     elif TYPE == 'choose_txt':
#         gen_func = gen_data_from_choose_txt
#     elif TYPE == 'fill_in_blank':
#         gen_func = gen_data_from_fill_in_blank
#     else:
#         raise ValueError('Unknown TYPE')

#     data = gen_func(counting_fols)
#     with open(f'test_skill_split/{SPLIT}_{TYPE}_{SKILL}.json', 'w') as f:
#         f.write('\n'.join([json.dumps(x) for x in data]))


# --- For generate training set ---

data = gen_data_from_choose_img(data_folders)
with open(f'{SPLIT}_captioned_iconqa_{TYPE}_with_ids.json', 'w') as f:
    f.write('\n'.join([json.dumps(x) for x in data]))
