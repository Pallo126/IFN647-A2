import os
import math
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pandas as pd

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

def process_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize the text and convert to lower case.
    filtered_tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords from the tokens.
    stemmer = PorterStemmer()  # Create an instance of PorterStemmer.
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]  # Apply stemming to each token.
    return stemmed_tokens

def load_queries(query_file_path):
    queries = {}
    with open(query_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
        for raw_query in raw_queries:
            number = re.search(r'<num> Number: (R\d+)', raw_query).group(1)
            title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
            queries[number] = process_text(title)
    return queries

def load_documents(directory_path):
    documents = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf8') as file:
            doc_id = filename.split('.')[0]
            documents[doc_id] = process_text(file.read())
    return documents

def calculate_bm25(N, avgdl, documents, queries, df):
    k1 = 1.2
    k2 = 500
    b = 0.75
    scores = {query_id: {} for query_id in queries}
    for query_id, query in queries.items():
        for doc_id, doc in documents.items():
            score = 0
            dl = len(doc)
            for word in set(query):
                if word in doc:
                    n = df.get(word, 0)
                    f = doc.count(word)
                    qf = query.count(word)
                    K = k1 * ((1 - b) + b * (dl / avgdl))
                    idf = math.log((N - n + 0.5) / (n + 0.5), 10)
                    term_score = idf * ((f * (k1 + 1)) / (f + K)) * ((qf * (k2 + 1)) / (qf + k2))
                    score += term_score
            scores[query_id][doc_id] = score
    return scores

def save_scores(scores, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for query_id, doc_scores in scores.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        output_file_path = os.path.join(output_folder, f"BM25_{query_id}Ranking.dat")
        with open(output_file_path, 'w') as file:
            for doc_id, score in sorted_docs:
                file.write(f"{doc_id} {score}\n")

# Example usage
query_file_path = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\the50Queries.txt'
base_data_directory = 'C:\\Users\\samin\\Desktop\\IFN647\\Assignment 2\\Data_Collection-1\\Data_Collection'
output_folder = 'RankingOutputsNEW'

queries = load_queries(query_file_path)
all_scores = {}

for i in range(101, 151):
    data_directory = os.path.join(base_data_directory, f"Data_C{i}")
    documents = load_documents(data_directory)
    N = len(documents)
    avgdl = sum(len(doc) for doc in documents.values()) / N
    df = {}
    for doc in documents.values():
        for word in set(doc):
            df[word] = df.get(word, 0) + 1
    scores = calculate_bm25(N, avgdl, documents, queries, df)
    all_scores.update(scores)

save_scores(all_scores, output_folder)

