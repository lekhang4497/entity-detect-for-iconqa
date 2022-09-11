import glob
import json
import os
from tqdm import tqdm

IMG_PATTERN = f'/home/khangln/JAIST_DRIVE/WORK/IconQA/data/iconqa_data/iconqa/test/choose_img/*'
TXT_PATTERN = f'/home/khangln/JAIST_DRIVE/WORK/IconQA/data/iconqa_data/iconqa/test/choose_txt/*'
FILL_PATTERN = f'/home/khangln/JAIST_DRIVE/WORK/IconQA/data/iconqa_data/iconqa/test/fill_in_blank/*'
PID2SKILL = '/home/khangln/JAIST_DRIVE/WORK/IconQA/data/iconqa_data/pid2skills.json'
with open(PID2SKILL) as f:
    pid2skill = json.load(f)


def extract_info(data_pattern):
    data_folders = glob.glob(data_pattern)
    pid2info = {}
    for t in tqdm(data_folders):
        data_path = os.path.join(t, 'data.json')
        pid = os.path.basename(t)
        with open(data_path) as f:
            d = json.load(f)
            label = d['answer']
        pid2info[pid] = {
            'label': label,
            'skill': pid2skill[pid]
        }
    return pid2info


my_info = {}
my_info.update(extract_info(IMG_PATTERN))
my_info.update(extract_info(TXT_PATTERN))
my_info.update(extract_info(FILL_PATTERN))

with open('/home/khangln/JAIST_DRIVE/WORK/IconQA/my/eval/pid2info.json', 'w') as f:
    json.dump(my_info, f, indent=4)
