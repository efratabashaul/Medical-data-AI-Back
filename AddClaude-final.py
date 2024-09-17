import pickle
from transformers import AutoTokenizer
import random
import torch
from transformers import BertModel
from pymongo import MongoClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import streamlit as st
import os
import faiss
import cohere
import boto3
import anthropic
import ssl

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE


boto3_bedrock = boto3.client(service_name='bedrock-runtime',region_name='us-east-1')
co = cohere.Client('IEo7xZnWi8cPD7jzR2dpsw3cvdGjchG6RCKPko05')
# describe=""

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

random_seed = 42
random.seed(random_seed)

# Set a random seed for PyTorch (for GPU as well)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

# Use a Hebrew BERT model
model_name = "avichr/heBERT"
model = BertModel.from_pretrained(model_name, resume_download=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)


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

# def save_embedding(word_embeddings,topic):
#     client = MongoClient("mongodb://localhost:27017/")
#     db = client["halacha_embedding"]
#     embedded_texts_collection = db[f"{topic}_embedded_vectors"]
#     embedded_texts_collection.insert_many([{"embedding": we.tolist()} for we in word_embeddings])

def save_embedding(word_embeddings):
  filename = "vectors.pkl"
  with open(filename, 'wb') as f:
    pickle.dump(word_embeddings, f)   

def load_file_content(filename):
    with open(filename, "r", encoding="utf-8") as file:
        content = file.readlines()
    return content

def process_sentences(sentences,topic):
    word_embeddings = embeddingVectors([doc["text"] for doc in sentences])
    save_embedding(word_embeddings,topic)



def DataPreprocessing(content, topic):
    current_title = None
    
    current_section = None
    sections = []

    section_letters = ["א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י", "יא", "יב", "יג", "יד", "טו", "טז", "יז", "יח", "יט", "כ", "כא", "כב", "כג", "כד", "כה", "כו", "כז", "כח", "כט", "ל"]

    for line in content:
        line = line.strip()
        line = re.sub(r'\[[^\]]*\]', '', line)
        line = re.sub(r'\([^)]*\)', '', line)
        line = re.sub(r'\.\s\.', '.', line)

        if line.startswith("!"):
            current_title = line[1:].strip() + "."
        elif any(line.startswith(letter) for letter in section_letters):
            if current_section:
                sections.append(current_section)
            current_section = {"title": current_title, "content": line[1:].strip()}
        else:
            if current_section:
                current_section["content"] += " " + line.strip()

    if current_section:
        sections.append(current_section)

    processed_texts = []
    for idx, section in enumerate(sections):
        chunks = text_splitter.split_text(section["content"])
        for chunk in chunks:
            document = {
                "metadata": {"title": section["title"], "subIndex": idx},
                "text": section["title"] + " " + chunk
            }
            processed_texts.append(document)
  
    process_sentences(processed_texts,topic)

    # שמירת הנתונים ב-MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["halacha"]
    processed_texts_collection = db[f"{topic}_processed_texts"]
    sections_collection = db[f"{topic}_sections"]

    processed_texts_collection.insert_many(processed_texts)
    sections_collection.insert_many(sections)




# def load_data_from_mongodb(topic):
#     client = MongoClient("mongodb://localhost:27017/")
#     db = client["halacha"]
#     processed_texts_collection = db[f"{topic}_processed_texts"]
#     sections_collection = db[f"{topic}_sections"]

#     processed_texts = list(processed_texts_collection.find())
#     sections = list(sections_collection.find())

#     return processed_texts, sections

def load_vectors_from_mongodb(topic):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["halacha_embedding"]
    embedded_texts_collection = db[f"{topic}_embedded_vectors"]

    embeddings = list(embedded_texts_collection.find({}, {"_id": 0, "embedding": 1}))

    return [torch.tensor(e["embedding"]) for e in embeddings]




def rerank_documents(query, docs, top_n):
    results = co.rerank(model="rerank-multilingual-v3.0", query=query, documents=docs, top_n=top_n, return_documents=True)
    print("Rerank results:", results)
    reranked_docs = [res.document.text for res in results.results]
    return reranked_docs

def find_sections_by_title(sections, title):
    return [section["content"] for section in sections if section["title"] == title]

def RAG(question, topic):
    encoded_sentences = load_vectors_from_mongodb(topic)
    encoded_sentences_tensor = torch.stack(encoded_sentences)  # המרת הרשימה לטנסור אחד

    query = [question]
    embedded_query = embeddingVectors(query)

    numpy_embeddings = encoded_sentences_tensor.cpu().numpy()
    index = faiss.IndexFlatL2(encoded_sentences_tensor.shape[1])
    index.add(numpy_embeddings)

    D, I = index.search(embedded_query.cpu().numpy(), k=20)

    processed_texts, sections = load_data_from_mongodb(topic)
    nearest_neighbors_texts = [processed_texts[idx]["text"] for idx in I[0]]

    reranked_texts = rerank_documents(query[0], nearest_neighbors_texts, top_n=20)
    first_rerank=""
    if reranked_texts:
        first_text = reranked_texts[0]
        first_period_index = first_text.find('.')
        if first_period_index != -1 and first_period_index <= 50:
            first_rerank = first_text[first_period_index+1:].strip()
        else:
            first_rerank = first_text

        
    processed_texts_2 = []
    for text in reranked_texts:
        first_period_index = text.find('.')
        if first_period_index != -1 and first_period_index <= 50:
            title = text[:first_period_index+1].strip()
            relevant_sections = find_sections_by_title(sections, title)
            combined_content = " ".join(relevant_sections)
            processed_texts_2.append(combined_content)
        else:
            processed_texts_2.append(text)

    top_3_texts = processed_texts_2[:3]

    context = " ".join(top_3_texts)

    client = anthropic.Client(api_key='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')

    response = client.messages.create(
        model="claude-2.1",
        system="Human: אתה רב שעונה על שאלות בהלכה. תסכם בשלוש-ארבע נקודות על פי המידע המצורף שבו יש תשובה לשאלה , אם אתה לא יודע, תכתוב איני יודע.",
        messages=[
            {"role": "user", "content": " Context: " + context + " Question: " + query[0]}
        ],
        max_tokens=400
    )

    answer = ""
    for text_block in response.content:
        answer += text_block.text
    return answer,first_rerank

# פונקציה לטעינת נושאים ממסד הנתונים
def loadTopics():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["halacha"]
    topics_collection = db["topics"]
    topics = list(topics_collection.find({}))
    return [topic["name"] for topic in topics]

# פונקציה לשמירת נושא חדש במסד הנתונים
def saveTopic(topic):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["halacha"]
    topics_collection = db["topics"]
    topics_collection.insert_one({"name": topic})



# ממשק המשתמש של Streamlit
st.image('logo.png', use_column_width=True)

# כותרת במרכז
st.markdown("<h1 style='text-align: center;'>שאל אותי כל שאלה</h1>", unsafe_allow_html=True)

if 'topics' not in st.session_state:
    st.session_state['topics'] =loadTopics()

selected_topic = st.selectbox("בחר נושא", st.session_state['topics'])

# שדה קלט עם placeholder
question = st.text_input('', placeholder='הכנס את השאלה כאן', key='input', label_visibility='collapsed')

# עמודות למרכז הכפתור
col1, col2, col3 = st.columns([1, 0.5, 1])

with col2:
    submit_button = st.button('שלח')

if submit_button:
    if question:
        # יצירת חלק ריק להצגת "LOADING"
        placeholder = st.empty()
        placeholder.text("LOADING")

        # שליחת השאלה והנושא לפונקציה RAG
        answer, source  = RAG(question, selected_topic)  # יש להחליף בקוד שמביא תשובות אמיתיות

        # החלפת "LOADING" בתוצאה
        placeholder.text("")
        # st.write(f'השאלה שלך: {question}')
        st.markdown(f"<div style='direction: rtl; text-align: right;'>מקור: {source}</div> </br>", unsafe_allow_html=True)
        st.markdown(f"<div style='direction: rtl; text-align: right;'>תשובת מענה RAG: {answer}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='direction: rtl; text-align: right;'>אנא הכנס שאלה.</div>", unsafe_allow_html=True)

# תיבת העלאת קבצים
uploaded_file = st.file_uploader("העלה קובץ טקסט", type=["txt"])

if uploaded_file:
    # הוספת שדה קלט לשם הנושא
    new_topic = st.text_input("הכנס שם לנושא החדש")

    if new_topic:
        # קריאת תוכן הקובץ
        content = uploaded_file.read().decode("utf-8")
        # פיצול לשורות
        lines = content.split("\n")
        # עיבוד השורות
        processed_lines = []
        for line in lines:
            if line.startswith("סימן"):
                line = "!" + line
            processed_lines.append(line)

        # עיבוד ושמירה של הנתונים
        DataPreprocessing(processed_lines, new_topic)
        # הוספת הנושא החדש לרשימת הנושאים
        saveTopic(new_topic)
        st.success(f"הנושא '{new_topic}' נוסף בהצלחה!")
        st.experimental_rerun()  
