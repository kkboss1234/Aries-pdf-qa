
# Pdf Answering AI 😃🌟

In the rapidly evolving field of artificial intelligence and natural language processing, the ability to extract meaningful information from documents is becoming increasingly crucial. The "PDF Question Answering AI" project aims to address this need by leveraging the capabilities of pre-trained transformer models to provide accurate and contextually relevant answers to user queries based on the content of PDF documents


## Table of Contents 💡
#### -> Overview
#### -> KeyFeatures
#### -> Approach
#### -> Methodology
#### -> Results and Conclusions
#### -> How to run ?
#### -> References
#### -> More Knowledge 
## Overview ⏳
Our project utilizes the BERT (Bidirectional Encoder Representations from Transformers) model, a state-of-the-art transformer architecture, fine-tuned on the SQuAD (Stanford Question Answering Dataset) to understand and generate human-like responses. The primary objective is to enable users to upload a PDF document, ask specific questions related to the content, and receive precise answers in real-time.
## Key Features 🎲
1)  PDF Text Extraction: Efficiently extracts text from PDF documents, preserving the structure and context to ensure accurate information retrieval.

2)  Question Processing: Utilizes advanced natural language processing techniques to interpret and understand user queries.

3) Answer Generation: Generates precise and contextually accurate answers by leveraging the fine-tuned BERT model, ensuring high relevance and accuracy.


## Approach 🎩
To address this problem, I adopted a multi-step approach involving several key stages: PDF text extraction, question processing, answer generation, and user interface development. My approach leverages the BERT (Bidirectional Encoder Representations from Transformers) model fine-tuned on the SQuAD (Stanford Question Answering Dataset) for its robust question-answering capabilities.


## Methodology
### 1. Data Preparation

    PDF Text Extraction:
        Utilized the PyMuPDF library for accurate text extraction from PDF documents. This ensured preservation of document structure and content context.

    Pre-processing:
        Extracted text was pre-processed to remove unnecessary formatting, ensuring it was formatted appropriately for input into the BERT model.

### 2. Model Selection

    BERT Model:
        Selected the 'bert-large-uncased-whole-word-masking-finetuned-squad' model from Hugging Face's Transformers library.
        This model is fine-tuned on the SQuAD dataset, making it highly effective for question-answering tasks.

    Tokenizer:
        Utilized the BERT tokenizer to convert text into token IDs suitable for input into the BERT model.

### 3. Question Answering

    Encoding:
        Concatenated the question and the passage (extracted text from PDF), then encoded them into input IDs using the BERT tokenizer.

    Segmentation:
        Created segment IDs to distinguish between the question and the passage.

    Model Inference:
        Fed the encoded inputs into the BERT model to obtain start and end scores, which indicate the positions of the answer in the passage.

    Answer Extraction:
        Extracted tokens corresponding to the highest start and end scores and concatenated them to form the final answer.

### 4. Handling Long Documents

    Chunking:
        Divided long documents into manageable chunks to ensure the input size did not exceed the model's maximum token limit.
        Each chunk was processed independently through the question-answering pipeline.

    Best Answer Selection:
        For each chunk, repeated the question-answering process and selected the best answer based on the highest combined start and end scores.

### 5. User Interface Development

    Flask for Deployment:
        Transitioned to Flask for a scalable and production-ready solution.
        Developed a robust backend API using Flask to handle PDF uploads and process user questions efficiently.

## Results and Conclusion 🌻
### Result 
The "PDF Question Answering AI" project successfully demonstrates the ability to extract relevant information from PDF documents and answer questions based on the extracted text. The system was tested on a variety of PDF documents and questions to evaluate its performance, accuracy, and robustness. Here, we present a detailed analysis of the results and discuss their significance and the insights gained.
### Conclusion
The "PDF Question Answering AI" project successfully showcases the potential of advanced NLP models like BERT for extracting and utilizing information from PDF documents. The results highlight both the strengths and limitations of the current approach, providing valuable insights for future enhancements. The system's ability to accurately answer questions based on PDF content demonstrates its practical utility and opens up avenues for further research and development in this area.

## How to run on your computer ?
## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/kkboss1234/Aries-pdf-qa.git
    ```
2. Navigate to the project directory:
    ```bash
    cd ./Aries-pdf-qa/
    next line:
    cd ./app/
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt

    ```
## Deploy on your local host:
```bash
    python run.py

```
After that a link will be displayed on your terminal , open it and then pdf-answering system will be up on your browser.
## References
- https://arxiv.org/pdf/1805.08092.pdf
- https://ieeexplore.ieee.org/abstract/document/9079274
- https://arxiv.org/pdf/1707.07328.pdf
- https://arxiv.org/pdf/1810.04805.pdf
- https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
- https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html
- https://nlp.seas.harvard.edu/2018/04/03/attention.html
## More Knowledge
For more information about my project you can view the project report created by me :
https://drive.google.com/file/d/16pjKKNka2VCwf_6uxqo0FdF8TAXpQkx-/view?usp=sharing
