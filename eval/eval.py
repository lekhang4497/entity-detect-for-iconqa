import glob
import json
import os
from tqdm import tqdm


BASELINE_PREDICT = '/home/khangln/JAIST_DRIVE/WORK/IconQA/results/choose_txt/exp_patch_transformer_ques_bert.json'
PID_TO_INFO = '/home/khangln/JAIST_DRIVE/WORK/IconQA/my/eval/pid2info.json'

with open(BASELINE_PREDICT) as f:
    pid2pred = json.load(f)['results']

with open(PID_TO_INFO) as f:
    pid2info = json.load(f)


def eval_skill(eval_skill=None):
    correct_count = 0
    total_count = 0

    for pid, pred in pid2pred.items():
        skills = pid2info[pid]['skill']
        if eval_skill is not None and eval_skill not in skills:
            continue
        total_count += 1
        label = pid2info[pid]['label']
        if pred == label:
            correct_count += 1
    if total_count == 0:
        return
    acc = round(correct_count / total_count * 100, 2)
    print(f'{eval_skill} - {acc}')
    # print('Eval skill:', eval_skill)
    # print('Accuracy:', acc)
    # print(f'Correct: {correct_count}/{total_count}')


all_skills = set([x for v in pid2info.values() for x in v['skill']])
print(all_skills)

for s in all_skills:
    eval_skill(s)
eval_skill()
