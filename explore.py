from transformers import AutoTokenizer, pipeline, BertModel
import random
import torch
import pickle
import re
import os
import faiss
import numpy as np
import cohere
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
import ssl
import re
from PersonDetailClass import PersonDetails
from dateutil import parser
from datetime import datetime
from dateutil.parser import ParserError
from dotenv import load_dotenv


load_dotenv()

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
print("os.getenv('COHERE_API_KEY')")
print(os.getenv('COHERE_API_KEY'))
co = cohere.Client( os.getenv('COHERE_API_KEY'))

os.environ['CURL_CA_BUNDLE'] = ''

# MARKDOWN separators as provided
MARKDOWN_SEPARATORS = [
   # "\n#{1,6} ",  # Headers
   # "```\n",  # Code blocks
   # "\n\\*\\*\\*+\n",  # Horizontal line (***)
   # "\n---+\n",  # Horizontal line (---)
  #  "\n___+\n",  # Horizontal line (___)
   "\n\n",  # Paragraph breaks
    "\n",  # Line breaks
    "\.",  # Periods
    " ",  # Spaces between words
]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

model_name = "avichr/heBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name, resume_download=True)


def process_lines(content):
    current_title = None
    current_section = None
    sections = []
    part_counter = 1  # נתחיל את המנייה של הכותרות מ-1
    if isinstance(content, str):
        content = content.splitlines()  # מפצל לפי \n
    current_title = f"PART{part_counter}"
    current_section = {"title": current_title, "content": ""}
    part_counter += 1

    for line in content:
        line = line.strip()
        line = re.sub(r'\.\s\.', '.', line)  # מחבר נקודות רצופות

        if not line:  # זיהוי כותרת לפי שורה ריקה
            if current_section and current_section["content"]:
                sections.append(current_section)
            current_title = f"PART{part_counter}"
            part_counter += 1
            current_section = {"title": current_title, "content": ""}
        else:
            if current_section:
                current_section["content"] += " " + line.strip()  # מוסיף את השורה לתוכן

    # בסיום, נוודא שהחלק האחרון מתווסף
    if current_section and current_section["content"]:
        sections.append(current_section)

    return sections

def process_sections(sections):
    processed_texts = []
    for idx, section in enumerate(sections):
        chunks = text_splitter.split_text(section["content"])
        for chunk in chunks:
            document = {
                "metadata": {"title": section["title"], "subIndex": idx},
                "text":  chunk
            }
            processed_texts.append(document)
    for i, part in enumerate(processed_texts, 1):
        print(f"Part {i}: {part}")
    return processed_texts




def embeddingVectors(texts):
    # מוודאים שאנחנו מעבירים רק את הטקסטים עצמם לפונקציה
    print("text")
    print(texts)
    encoding = tokenizer.batch_encode_plus(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=True
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        word_embeddings = outputs.last_hidden_state
        sentence_embedding = word_embeddings[:, 0, :]  # Embedding של המשפט כולו
    return sentence_embedding


def process_sentences(sentences):
    if isinstance(sentences, list) and all(isinstance(s, dict) and "text" in s for s in sentences):
        sentences = [item["text"] for item in sentences]  # שליפת השדה 'text' מתוך כל אובייקט
        word_embeddings = embeddingVectors(sentences)
        save_embedding(word_embeddings)
    else:
        print("Error: sentences should be a list of dictionaries containing 'text' fields.")
def save_embedding(word_embeddings):
    filename = "vectors.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(word_embeddings, f)

def rerank_documents(query, docs, top_n):
    results = co.rerank(model="rerank-multilingual-v3.0", query=query, documents=docs, top_n=top_n, return_documents=True)
    print("Rerank results:", results)  # הדפס את התוצאות עבור בדיקה
    reranked_docs = [res.document.text for res in results.results]
    print("Reranked documents:", reranked_docs)  # הדפס את המסמכים לאחר הרירנק
    return reranked_docs

def load_encoded_sentences(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            encoded_sentences = pickle.load(f)
            return encoded_sentences
    return None

def process_query(query, encoded_sentences,processed_texts):
    embedded_query = embeddingVectors([query])
    numpy_embeddings = encoded_sentences.reshape(-1, encoded_sentences.shape[1]).cpu().numpy()
    index = faiss.IndexFlatL2(encoded_sentences.shape[1])
    index.add(numpy_embeddings)
    print("encoded sentences: ")
    print(numpy_embeddings)
    D, I = index.search(embedded_query.cpu().numpy(), k=10)
    nearest_neighbors_texts = [processed_texts[idx]["text"] for idx in I[0]]
    print("Nearest neighbors texts:", nearest_neighbors_texts)  # הדפס את הטקסטים הקרובים ביותר
    reranked_texts = rerank_documents(query, nearest_neighbors_texts, top_n=20)
    return reranked_texts

def create_context(processed_texts, top_n=3):
    top_texts = processed_texts[:top_n]
    context = " ".join(top_texts)
    return context


def main(query,text):
    sections = process_lines(text)
    processed_text = process_sections(sections)
    process_sentences(processed_text)
    filename = 'vectors.pkl'
    encoded_sentences = load_encoded_sentences(filename)
    if encoded_sentences is not None:
        reranked_texts = process_query(query, encoded_sentences, processed_text)
        context = create_context(reranked_texts, top_n=5)
        oracle = pipeline('question-answering', model='dicta-il/dictabert-heq')
        # נבצע את השאילתות עבור כל הפרטים הדרושים
        return oracle(question=query, context=context)['answer']
    return None



def parse_date(date_string):
    try:
        print("kkkk")
        cleaned_string = re.sub(r'[^0-9/-]', '', date_string)
        print(cleaned_string)
        parsed_date = parser.parse(cleaned_string)
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError:
        return ''
       # raise ValueError(f"Cannot parse date: {date_string}")


#def parse_date(date_string):
 #   date_string = clean_date_string(date_string)
  #  formats = ["%a, %d %b %Y %H:%M:%S %Z","%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y","%Y/%m/%d","%m-%d-%Y","%Y.%m.%d","%m.%d.%Y","%d.%m.%Y"]  # הוסיפי פורמטים נוספים לפי הצורך
   # for fmt in formats:
    #    try:
     #       parsed_date = datetime.strptime(date_string, fmt)
      #      # אם ההמרה הצליחה, נחזיר את התאריך בפורמט המבוקש: 02-03-2024
       #     return parsed_date.strftime("%Y-%m-%d")
        #except ValueError:
         #   continue
    #return date_string
def arrange(text):
    #name_answer = ''

    name_answer = main("מה שם הפצוע?", text)
    #age_answer = ''

    age_answer = main("בן כמה הפצוע?", text)
    #name_father_answer = ''
    name_father_answer = main("מה שם האבא של הפצוע?", text)

    id_number_answer = main("מה ת.ז. הפצוע?", text)
    hospital_answer = main("באיזה בית חולים הפצוע?", text)
    date_of_injury_answer = main("מתי הייתה הפציעה?", text)
    print("date_of_injury_answer")
    print(date_of_injury_answer)
    date_of_injury_answer=parse_date(date_of_injury_answer)
    doctor_answer = main("מי הרופאים המטפלים?", text)
    # יצירת אובייקט מסוג PersonDetails עם כל הפרטים שהתקבלו
    if(date_of_injury_answer==''):
        date_of_injury_answer=None
    person_details = PersonDetails(
        name=name_answer,
        age=age_answer,
        id=id_number_answer,
        hospital=hospital_answer,
        date=date_of_injury_answer,
        doctorType=doctor_answer,
        nameFather=name_father_answer
    )

    # החזרת אובייקט PersonDetails
    return person_details

if __name__ == "__main__":
    arrange()

#שם-אחמד
#שם האבא-מוחמד
#ת.ז.-הביא טוב
#גיל-כשכתבתי בן ושאלתי גיל-הביא תז, אך כששאלתי בן -הביא טוב!, וכשכתבתי גיל-ושאלתי גיל וגם כששאלתי בן-הביא טוב-לכן תמיד לכתוב בן כמה!!
#בית חולים-גם טוב
#תאריך-כששאלתי מתי התאריך-וכתבתי את המילה תאריך-הביא טוב
#אך כשלא כתבתי את המילה תאריך-ושאלתי תאריך-הביא תז,אך כששאלתי מתי-הביא טוב!!
#"מי הרופאים המטפלים?-הביא רופא משפחה
#כששאלתי "מה הבית חולים שבו נמצא הפצוע?" ענה רופא משפחה
#כשלא  ירדתי בין השורות-שם,גיל, ובית חולים -הביא מעולה-
