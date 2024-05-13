import os
import re
import math
from collections import defaultdict, Counter

# Constants
k1 = 1.2
k2 = 500
b = 0.75

# Paths
data_dir = 'C:/Users/pallavi/Downloads/ifn647_ass2_docs/Data_Collection-1/Data_Collection'
query_file_path = 'C:/Users/pallavi/Downloads/ifn647_ass2_docs/the50Queries.txt'
output_dir = 'C:/Users/pallavi/Downloads/ifn647_ass2_docs/RankingOutputs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Helper Functions
def tokenize(text):
    """ Simple tokenizer to split and lower case the text. """
    return re.findall(r'\b\w+\b', text.lower())

def load_queries(filename):
    """ Loads and parses queries from a file. """
    queries = {}
    with open(filename, 'r') as f:
        text = f.read()
    matches = re.findall(r'<top>(.*?)</top>', text, re.DOTALL)
    for match in matches:
        num = re.search(r'<num> Number: R(\d+)', match)
        title = re.search(r'<title> (.*)', match)
        if num and title:
            queries[num.group(1)] = tokenize(title.group(1))
    print(f"Loaded {len(queries)} queries.")
    return queries

def load_documents(directory):
    """ Loads all documents from subdirectories and tokenizes them. """
    docs = defaultdict(dict)
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                with open(os.path.join(folder_path, filename), 'r') as file:
                    docs[folder][filename] = tokenize(file.read())
            print(f"Loaded documents from {folder}: {len(docs[folder])} files.")
    return docs

def bm25(n, f, qf, N, dl, avdl):
    """ Calculate BM25 for a single term in a document. """
    K = k1 * ((1 - b) + b * (dl / avdl))
    return max(math.log((N - n + 0.5) / (n + 0.5)) * ((k1 + 1) * f) / (K + f) * ((k2 + 1) * qf) / (k2 + qf), 0)

# Load data
queries = load_queries(query_file_path)
all_docs = load_documents(data_dir)

# Process each query
for query_num, query_tokens in queries.items():
    dataset_id = f"Data_C{int(query_num) + 100}"  # Ensure dataset ID matches folder names
    if dataset_id in all_docs:
        output_path = os.path.join(output_dir, f"BM25_R{query_num}Ranking.dat")
        print(f"Attempting to write results to {output_path}")
        with open(output_path, 'w') as outfile:
            doc_scores = []
            for doc_id, doc_tokens in all_docs[dataset_id].items():
                score = 0
                doc_len = len(doc_tokens)
                avg_doc_len = sum(len(tokens) for tokens in all_docs[dataset_id].values()) / len(all_docs[dataset_id])
                for term in query_tokens:
                    n = sum(term in tokens for tokens in all_docs[dataset_id].values())
                    f = doc_tokens.count(term)
                    qf = query_tokens.count(term)
                    score += bm25(n, f, qf, len(all_docs[dataset_id]), doc_len, avg_doc_len)
                if score > 0:  # Only consider documents with a positive score
                    doc_scores.append((doc_id, score))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            for doc_id, score in doc_scores:
                outfile.write(f"{doc_id}\t{score}\n")
            print(f"Successfully written to {output_path}")
    else:
        print(f"No matching dataset found for query {query_num}")

print("All processing complete.")
