from transformers import AutoTokenizer, pipeline
import random
import torch
from transformers import BertModel
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os
import faiss
import numpy as np
import cohere
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.load.dump import dumps
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import json
import os
import sys
import boto3
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#import anthropic
import ssl

text="בתאריך 05-08-2000 אחמד נפצע, ת.ז.: 329857445 ,אחמד בן 5,שם אביו -מוחמד,כעת הוא נמצא בבית החולים 'שובא',ומטופל על ידי רופא משפחה."

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

boto3_bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-east-1')

co = cohere.Client('IEo7xZnWi8cPD7jzR2dpsw3cvdGjchG6RCKPko05')

os.environ['CURL_CA_BUNDLE'] = ''

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    ".",
    " ",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

model_name = "avichr/heBERT"
model = BertModel.from_pretrained(model_name, resume_download=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def load_file_content(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.readlines()
    return content


def process_lines(content):
    current_section = None
    sections = []

    for line in content:
        line = line.strip()
        line = re.sub(r'\[[^\]]*\]', '', line)  # הסרת תוכן בתוך סוגריים מרובעים
        line = re.sub(r'\([^)]*\)', '', line)   # הסרת תוכן בתוך סוגריים עגולים
        line = re.sub(r'\.\s\.', '.', line)     # תיקון נקודות מיותרות

        if line == "":  # שורה ריקה - סימן לפסקה חדשה
            if current_section:
                sections.append(current_section)
                current_section = None  # איפוס המקטע הנוכחי
        else:
            if current_section is None:
                current_section = {"content": line}  # התחלת פסקה חדשה
            else:
                current_section["content"] += " " + line.strip()

    if current_section:
        sections.append(current_section)  # הוספת הפסקה האחרונה

    return sections


def process_sections(sections):
    processed_texts = []
    for idx, section in enumerate(sections):
        chunks = text_splitter.split_text(section["content"])
        for chunk in chunks:
        #     document = {
        #         "metadata": {"title": section["title"], "subIndex": idx},
        #         "text": section["title"] + " " + chunk
        #     }
            processed_texts.append(chunk)
    return processed_texts


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)





# def main2(filename):
#     content = load_file_content(filename)
#     sections = process_lines(content)
#     processed_texts = process_sections(sections)

#     set_random_seed(42)
#     model_name = "avichr/heBERT"
#     model, tokenizer = load_hebrew_bert_model(model_name)

#     return processed_texts, model, tokenizer


def process_sentences(sentences):
    word_embeddings = embeddingVectors( sentences)
    save_embedding(word_embeddings)


def embeddingVectors(text):
    encoding = tokenizer.batch_encode_plus(
        text,  # List of input texts
        padding=True,  # Pad to the maximum sequence length
        truncation=True,  # Truncate to the maximum sequence length if necessary
        return_tensors='pt',  # Return PyTorch tensors
        add_special_tokens=True  # Add special tokens CLS and SEP
    )
    input_ids = encoding['input_ids']  # Token IDs
    attention_mask = encoding['attention_mask']  # Attention mask

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state  # (batch_size, sequence_length, embedding_dim)
        # החזרת ה-embedding של ה-CLS token
        sentence_embedding = word_embeddings[:, 0, :]  # בחירת האיבר הראשון בכל רצף (CLS token)
    return sentence_embedding  # This contains the embeddings


def save_embedding(word_embeddings):
    filename = "vectors.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(word_embeddings, f)


def rerank_documents(query, docs, top_n):
    results = co.rerank(model="rerank-multilingual-v3.0", query=query, documents=docs, top_n=top_n,
                        return_documents=True)
    print("Rerank results:", results)  # הוספת הדפסה לבדיקת מבנה התוצאה
    reranked_docs = [res.document.text for res in results.results]  # עדכון בהתאם למבנה הנכון
    return reranked_docs


# פונקציה למציאת כל הסעיפים עם כותרת מסוימת
# def find_sections_by_title(sections, title):
#     return [section["content"] for section in sections if section["title"] == title]


def load_encoded_sentences(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            encoded_sentences = pickle.load(f)
            return encoded_sentences
    return None

def process_query(query, encoded_sentences):
    embedded_query = embeddingVectors([query])
    numpy_embeddings = encoded_sentences.reshape(-1, encoded_sentences.shape[1]).cpu().numpy()
    index = faiss.IndexFlatL2(encoded_sentences.shape[1])
    index.add(numpy_embeddings)
    D, I = index.search(embedded_query.cpu().numpy(), k=20)
    nearest_neighbors_texts = [text[idx] for idx in I[0]]
    reranked_texts = rerank_documents(query, nearest_neighbors_texts, top_n=20)
    return reranked_texts
 
# def trim_texts(reranked_texts):
#     processed_texts = []
#     for text in reranked_texts:
#         first_period_index = text.find('.')
#         if first_period_index != -1 and first_period_index <= 50:
#             title = text[:first_period_index + 1].strip()
#             relevant_sections = find_sections_by_title(sections, title)
#             combined_content = " ".join(relevant_sections)
#             processed_texts.append(combined_content)
#         else:
#             processed_texts.append(text)
#     return processed_texts

def create_context(processed_texts, top_n=3):
    top_texts = processed_texts[:top_n]
    context = " ".join(top_texts)
    return context

# def get_response_from_claude(context, query):
#     client = anthropic.Client(api_key="sk-ant-api03-UAA")
#     response = client.messages.create(
#         model="claude-2.1",
#         system="Human: אתה רב שעונה על שאלות בהלכה. תסכם בשלוש-ארבע נקודות על פי המידע המצורף שבו יש תשובה לשאלה , אם אתה לא יודע, תכתוב איני יודע.",
#         messages=[
#             {"role": "user", "content": " Context: " + context + " Question: " + query}
#         ],
#         max_tokens=200
#     )
#     return response

def print_response(response):
    for text_block in response.content:
        print(text_block.text)

def main(query):
    sections = process_lines(text)
    processed_texts = process_sections(sections)
    process_sentences(processed_texts)
    filename = 'vectors.pkl'
    # content = load_file_content(filename)
    encoded_sentences = load_encoded_sentences(filename)
    if encoded_sentences is not None:
        print(type(encoded_sentences))
        reranked_texts = process_query(query, encoded_sentences)

        # processed_texts = trim_texts(reranked_texts)
        # context = create_context(processed_texts, top_n=3)
        context = create_context(reranked_texts, top_n=3)
        oracle=pipeline('question-answering',model='dicta-il/dictabert-heq')
        print("response: "+oracle(question=query,context=context))
        # response = get_response_from_claude(context, query)
        # print_response(response)

if __name__ == "__main__":
    user_query = "מה הת.ז. של הפצוע?"
    main(user_query)