from segment import count_region_of_img
import json
import os
from tqdm import tqdm
import icon_classify

COUNTING_PATH = '/home/khangln/JAIST_DRIVE/WORK/IconQA/results/choose_img/exp_patch_transformer_ques_bert.json'
DATA_PATH = '/home/khangln/JAIST_DRIVE/WORK/IconQA/filtered_icon/icon_counting/test/choose_img'
TRAIN_DATA_PATH = '/home/khangln/JAIST_DRIVE/WORK/IconQA/filtered_icon/icon_counting_test.jsonl'

with open(COUNTING_PATH) as f:
    data = json.load(f)
total_len = len(data['results'])

icon_data = []
choice_len = []

c = 0
for r, data_dirs, _ in os.walk(DATA_PATH):
    for d in data_dirs:
        c += 1
        print(f'{c}/{total_len}')
        with open(os.path.join(r, d, 'data.json')) as f:
            data = json.load(f)
            if any([x in data['question'] for x in ['side', 'sides', 'corner', 'corners']]):
                continue
            choice_img_path = [os.path.join(r, d, x) for x in data['choices']]
            count = [count_region_of_img(x) for x in choice_img_path]
            icon_data.append({
                "question_id": d,
                "question": data['question'],
                "choices": count,
                "answer": data['answer']
            })
            choice_len.append(len(count))

print(choice_len, max(choice_len))

with open(TRAIN_DATA_PATH, 'w') as f:
    f.write('\n'.join(json.dumps(x) for x in icon_data))
