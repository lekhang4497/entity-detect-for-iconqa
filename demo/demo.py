from extract_entity import classify_all_entity, extract_bbox
from icon_classify import classify_img
from PIL import Image
import tokenizers
from skimage import io
from skimage.color import rgb2gray
from collections import defaultdict
import streamlit as st
import json
import numpy as np
import glob
import os
from PIL import Image
from bertviz import head_view
from transformers import BertTokenizer, BertForQuestionAnswering, AutoModelForMultipleChoice, AutoTokenizer
import torch
from streamlit import components
import sys
sys.path.append('../')


ICON_QA_DATA_DIR = '/home/khangln/JAIST_DRIVE/WORK/IconQA/data/iconqa_data/iconqa'

PROBLEM_DICT = 'problems_dict.json'

# TASK = 'choose_img'

DESC_MODEL = '/home/khangln/H_JAIST_DRIVE/WORK/iconqa_project/entity-detect-for-iconqa/models/bert-base-cased-train-both-img-txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@st.cache(hash_funcs={tokenizers.Tokenizer: id})
def get_desc_model():
    tokenizer = AutoTokenizer.from_pretrained(DESC_MODEL)
    model = AutoModelForMultipleChoice.from_pretrained(DESC_MODEL)
    model.to(device)
    return model, tokenizer


@st.cache()
def get_question_dict():
    st.write("Cache miss: get_question_dict() ran")
    q_dict = {}
    with open(PROBLEM_DICT) as f:
        problem_dict = json.load(f)
    for k, problem in problem_dict.items():
        if problem['split'] not in q_dict:
            q_dict[problem['split']] = {}
        split_dict = q_dict[problem['split']]
        if problem['ques_type'] not in split_dict:
            split_dict[problem['ques_type']] = [k]
        else:
            split_dict[problem['ques_type']].append(k)
    return q_dict


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
    # ret = []
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


def ent_img_to_sent(pis):
    return entities_to_sentence([classify_img(pi) for pi in pis])


def chunk(lst, size):
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def entities_to_sentence(entities):
    fq = defaultdict(int)
    for ent in entities:
        fq[ent] += 1
    return ', '.join([f'{v} {k}' for k, v in fq.items()])


def predict_multiple_choice(prompt, candidates):
    model, tokenizer = get_desc_model()
    inputs = tokenizer([[prompt, x] for x in candidates],
                       return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()})
        logits = outputs.logits
        return logits.argmax().item()


example_paths = glob.glob(os.path.join(ICON_QA_DATA_DIR, '*/*/*'))
id2path = {os.path.basename(p): p for p in example_paths}
st.title('Context Enrichment for Mathematics Abstract VQA')

split2display = {
    'train': 'Train',
    'val': 'Validation',
    'test': 'Test'
}

task2display = {
    'choose_img': 'Multiple choice Image',
    'choose_txt': 'Multiple choice Text',
    'fill_in_blank': 'Abstract text answer'
}

col1, col2 = st.columns(2)
with col1:
    subtask = st.selectbox('Choose task', [
        'choose_img', 'choose_txt', 'fill_in_blank'], format_func=lambda x: task2display[x])

with col2:
    split = st.selectbox('Choose data split', [
        'test', 'val',  'train'],  format_func=lambda x: split2display[x])

# filtered_questions = get_question_dict()[split][subtask]
# filtered_q_ids = [x['id'] for x in filtered_questions]

filtered_q_ids = get_question_dict()[split][subtask]

iconqa_qid = st.selectbox(
    'Choose an example', filtered_q_ids, format_func=lambda x: f'{x}: {problem_dict[x]["question"]}')

st.subheader('Question:')
image_path = os.path.join(id2path[iconqa_qid], 'image.png')
image = Image.open(image_path)
st.image(remove_border(image))

with open(os.path.join(id2path[iconqa_qid], 'data.json')) as f:
    q_data = json.load(f)

if subtask in ['choose_img', 'choose_txt']:
    st.subheader('Choices:')
    choice_cols = st.columns(len(q_data['choices']))
    for col, q_choice in zip(choice_cols, q_data['choices']):
        if subtask == 'choose_img':
            image = Image.open(os.path.join(id2path[iconqa_qid], q_choice))
            col.image(remove_border(image))
        else:
            col.info(f'**{q_choice}**')

col1, col2, col3 = st.columns(3)
with col2:
    perform_btn = st.button('Perform Method', type='primary')

if perform_btn:
    pis, im = classify_all_entity(image_path)
    st.subheader('Connected Component Labeling')
    segment_im = np.array(
        [[[255, 0, 0] if c == -1 else [255, 255, 255] for c in r] for r in im])
    st.image(Image.fromarray(segment_im.astype(np.uint8)))
    st.markdown("""---""")
    st.subheader('Detected Objects')
    with st.spinner('Detecting'):
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
        st.markdown("""---""")
        st.subheader('Context Enrichment')
        st.markdown(f'**Context Enrichment:** **:red[{context_enrich}]**')
        if subtask == 'choose_img':
            st.markdown(f'**Generated Textual Choices**')
            candidate_choices = []
            choice_cols = st.columns(len(q_data['choices']))
            for col, choice_img in zip(choice_cols, q_data['choices']):
                img_path = os.path.join(id2path[iconqa_qid], choice_img)
                # Generate textual choice
                pis, _ = classify_all_entity(img_path)
                textual_choice = ent_img_to_sent(pis)
                candidate_choices.append(textual_choice)
                col.markdown(f'**:red[{textual_choice}]**')
                # Show choice image
                image = Image.open(img_path)
                col.image(remove_border(image))
        elif subtask == 'choose_txt':
            candidate_choices = q_data['choices']

        st.markdown("""---""")
        st.subheader('Answer Prediction')
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown(f'**:red[Answer]**')
            prompt = f'{q_data["question"]} {context_enrich}'
            answer_idx = predict_multiple_choice(prompt, candidate_choices)
            if subtask == 'choose_img':
                img_path = os.path.join(
                    id2path[iconqa_qid], q_data['choices'][answer_idx])
                image = Image.open(img_path)
                st.image(remove_border(image))
            elif subtask == 'choose_txt':
                st.info(q_data['choices'][answer_idx])

    st.markdown("""---""")
    st.subheader('Attention Visualization')
    with st.spinner('Visualizing Attention'):
        model_version = '/home/khangln/H_JAIST_DRIVE/WORK/iconqa_project/entity-detect-for-iconqa/models/bert-base-cased-train-both-img-txt'
        model = BertForQuestionAnswering.from_pretrained(
            model_version, output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(model_version)
        # sentence_a = q_data['question']
        # sentence_b = context_enrich
        # inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
        # input_ids = inputs['input_ids']
        # token_type_ids = inputs['token_type_ids']
        # attention = model(input_ids, token_type_ids=token_type_ids)[-1]
        # sentence_b_start = token_type_ids[0].tolist().index(1)
        # input_id_list = input_ids[0].tolist() # Batch index 0
        # tokens = tokenizer.convert_ids_to_tokens(input_id_list)

        def visualize_info(sent1, sent2):
            inputs = tokenizer.encode_plus(sent1, sent2, return_tensors='pt')
            input_ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            attention = model(input_ids, token_type_ids=token_type_ids)[-1]
            sentence_b_start = token_type_ids[0].tolist().index(1)
            input_id_list = input_ids[0].tolist()  # Batch index 0
            tokens = tokenizer.convert_ids_to_tokens(input_id_list)
            att_shape = attention[0].shape
            my_att = [torch.broadcast_to(torch.sum(
                att, dim=1, keepdim=True), (1, 12, att_shape[2], att_shape[3])) for att in attention]
            return my_att, tokens, sentence_b_start

        html_obj = head_view(*visualize_info(q_data['question'], context_enrich + '[SEP]' + '[SEP]'.join(
            candidate_choices)), html_action='return', heads=[0])
        components.v1.html(html_obj._repr_html_(), height=500)
