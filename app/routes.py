from flask import Blueprint, render_template, request
import fitz  # PyMuPDF
from transformers import BertForQuestionAnswering, BertTokenizer
import torch
import numpy as np

main = Blueprint('main', __name__)

# Load pre-trained model and tokenizer
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

def extract_text_from_pdf(file_stream):
    # Open the provided PDF file
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)  # load a page
        text += page.get_text()  # extract text from page
    return text

def bert_question_answer(question, passage, max_len=512):
    input_ids = tokenizer.encode(question, passage, max_length=max_len, truncation=True, return_tensors="pt")
    sep_index = torch.where(input_ids == tokenizer.sep_token_id)[1].tolist()[0]
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids[0]) - num_seg_a

    segment_ids = torch.cat((torch.zeros(num_seg_a, dtype=torch.long),
                             torch.ones(num_seg_b, dtype=torch.long)))

    outputs = model(input_ids, token_type_ids=segment_ids.unsqueeze(0))
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    start_scores = start_scores.detach().numpy().flatten()
    end_scores = end_scores.detach().numpy().flatten()

    start_index = np.argmax(start_scores)
    end_index = np.argmax(end_scores)

    start_score = np.round(start_scores[start_index], 2)
    end_score = np.round(end_scores[end_index], 2)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    answer = tokens[start_index]
    for i in range(start_index + 1, end_index + 1):
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]
        else:
            answer += ' ' + tokens[i]

    if start_score < 0 or start_index == 0 or end_index < start_index or answer == '[SEP]':
        answer = "Sorry, I was unable to find an answer in the passage."

    return start_score, end_score, answer

def get_best_answer(question, text, max_len=512, chunk_size=400):
    all_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    best_answer = ""
    best_score = -float('inf')

    for chunk in all_chunks:
        start_score, end_score, answer = bert_question_answer(question, chunk, max_len=max_len)
        if start_score + end_score > best_score:
            best_score = start_score + end_score
            best_answer = answer

    return best_answer

@main.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        pdf_file = request.files['pdf']
        question = request.form['question']
        text = extract_text_from_pdf(pdf_file)
        answer = get_best_answer(question, text)
    return render_template('index.html', answer=answer)
