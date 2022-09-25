import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import evaluate

MODEL = '/home/khangln/H_JAIST_DRIVE/WORK/iconqa_project/entity-detect-for-iconqa/seq2seq_models/bart-large-train-fill-blank-2'
TEST_FILE = '/home/khangln/H_JAIST_DRIVE/WORK/iconqa_project/entity-detect-for-iconqa/data/fill_in_blank/test_captioned_iconqa_fill_in_blank.json'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
model.to(device)


def gen_answer(texts):
    if len(texts) == 0:
        return []
    global tokenizer
    global model

    loader = DataLoader(texts, batch_size=128)
    ret = []
    for batch in tqdm(loader):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True).to(device)
        outputs = model.generate(inputs["input_ids"])
        batch_summaries = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        ret.extend(batch_summaries)

    return ret


questions = []
labels = []
with open(TEST_FILE) as f:
    for line in f:
        d = json.loads(line)
        questions.append(d['source'])
        labels.append(d['target'])


answers = gen_answer(questions)

print('Accuracy')

metric = evaluate.load("exact_match")
results = metric.compute(predictions=answers, references=labels)
print(results)

# for a, l in zip(answers, labels):
#     print(a, ' --- ', l)
# correct = sum([1 if a == l else 0 for a, l in zip(answers, labels)])
# print(round(correct / len(questions) * 100, 2))
