from PIL import Image, ImageOps
from skimage import io
from skimage.color import rgb2gray
from collections import defaultdict
from tqdm import tqdm
from typing import List
import streamlit as st
import pandas as pd
import json
import numpy as np
import glob
import os
from PIL import Image
import sys
sys.path.append('../')
from extract_entity import classify_all_entity, bfs, extract_bbox
from icon_classify import classify_img



ICON_QA_DATA_DIR = '/home/khangln/JAIST_DRIVE/WORK/IconQA/data/iconqa_data/iconqa/test'

PROBLEM_DICT = 'problems_dict.json'

TASK = 'choose_img'

with open(PROBLEM_DICT) as f:
    problem_dict = json.load(f)


def remove_border(im):
    border_width = 2
    width, height = im.size
    left = border_width
    top = border_width
    right = width - border_width
    bottom = height - border_width

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def classify_all_entity(img_path, min_pixels=100):
    ret = []
    img = io.imread(img_path)
    img = img[3:-3, 3:-3]
    img[np.logical_and(img[:, :, 0] == 178, img[:, :, 1] ==
                       235, img[:, :, 2] == 255)] = [255, 255, 255]
    gimg = rgb2gray(img)
    # Set almost white pixel to white
    # gimg[gimg > 0.85] = 1.0

    bboxs = extract_bbox(gimg)
    bboxs = [b for b in bboxs if abs(b[2]-b[0]+1)*abs(b[3]-b[1]) >= min_pixels]
    pis = []
    for b in bboxs:
        top, left, bottom, right = b
        pi = Image.fromarray(img[top:bottom+1, left:right+1])
        # entity_class = classify_img(pi)
        # ret.append(entity_class)
        pis.append(pi)
    return pis, gimg


example_paths = glob.glob(os.path.join(ICON_QA_DATA_DIR, TASK, '*'))
id2path = {os.path.basename(p): p for p in example_paths}
st.title('Context Enrichment for Mathematics Abstract VQA')

iconqa_qid = st.selectbox(
    'Choose an example', id2path.keys(), format_func=lambda x: f'{x}: {problem_dict[x]["question"]}')

st.subheader('Question:')
image_path = os.path.join(id2path[iconqa_qid], 'image.png')
image = Image.open(image_path)
st.image(remove_border(image))

with open(os.path.join(id2path[iconqa_qid], 'data.json')) as f:
    q_data = json.load(f)

st.subheader('Choices:')
choice_cols = st.columns(len(q_data['choices']))
for col, choice_img in zip(choice_cols, q_data['choices']):
    image = Image.open(os.path.join(id2path[iconqa_qid], choice_img))
    col.image(remove_border(image))


def chunk(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]

def entities_to_sentence(entities):
    fq = defaultdict(int)
    for ent in entities:
        fq[ent] += 1
    return ', '.join([f'{v} {k}' for k, v in fq.items()])

if st.button('Perform Method', kwargs={'type':'primary'}):
    pis, im = classify_all_entity(image_path)
    st.subheader('Connected Component Labeling')
    segment_im = np.array(
        [[[255, 0, 0] if c == -1 else [255, 255, 255] for c in r] for r in im])
    st.image(Image.fromarray(segment_im.astype(np.uint8)))
    st.subheader('Detected Objects')
    pis_chunks = chunk(pis, 5)
    entities = []
    for pis_chunk in pis_chunks:
        pis_cols = st.columns(5)
        for col, pi in zip(pis_cols, pis_chunk):
            pi_class = classify_img(pi)
            entities.append(pi_class)
            col.image(pi)
            col.caption(pi_class)
    context_enrich = entities_to_sentence(entities)
    st.markdown(f'**Context Enrichment:** **:red[{context_enrich}]**')
