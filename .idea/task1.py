import os
import re
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download necessary NLTK packages if not already present
nltk.download('punkt')  # Tokenizers for various languages.
nltk.download('stopwords')  # Stopwords like 'and', 'the', etc., which are not useful for search.

# Load stop words from NLTK library once at the start to improve efficiency.
stop_words = set(stopwords.words('english'))

def process_text(text):
    """
    Processes the input text by lowering the case, tokenizing, removing stopwords, and applying stemming.
    Args:
        text (str): The text to process.
    Returns:
        list: A list of stemmed tokens.
    """
    # Tokenize the text and convert each word to lower case
    tokens = word_tokenize(text.lower())
    # Remove stopwords to reduce dataset size and improve focus on meaningful words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Stemming words to reduce them to their root form
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

def load_queries(query_file_path):
    """
    Loads and processes queries from a file.
    Args:
        query_file_path (str): The file path for the query file.
    Returns:
        dict: A dictionary of queries where each key is the query ID and value is the list of processed tokens.
    """
    queries = {}
    with open(query_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Regex to find queries enclosed in <Query> tags
        raw_queries = re.findall(r'<Query>(.*?)</Query>', content, re.DOTALL)
        for raw_query in raw_queries:
            # Extract query number and title
            number = re.search(r'<num> Number: (R\d+)', raw_query).group(1)
            title = re.search(r'<title>(.*?)\n', raw_query).group(1).strip()
            # Process and store the query text
            queries[number] = process_text(title)
    return queries

def load_documents(directory_path):
    """
    Loads and processes all documents from a directory.
    Args:
        directory_path (str): Directory containing document files.
    Returns:
        dict: A dictionary of documents where each key is the document ID and value is the list of processed tokens.
    """
    documents = {}
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf8') as file:
            doc_id = filename.split('.')[0]
            documents[doc_id] = process_text(file.read())
    return documents

def calculate_bm25(N, avgdl, documents, queries, df):
    """
    Calculates the BM25 score for each document against each query.
    Args:
        N (int): Total number of documents.
        avgdl (float): Average length of documents.
        documents (dict): Dictionary of documents.
        queries (dict): Dictionary of queries.
        df (dict): Document frequency of each word.
    Returns:
        dict: Dictionary of dictionaries containing BM25 scores for each document against each query.
    """
    k1 = 1.2
    k2 = 500
    b = 0.75
    scores = {query_id: {} for query_id in queries}
    for query_id, query in queries.items():
        for doc_id, doc in documents.items():
            score = 0
            dl = len(doc)
            # Calculate score for each word in the query present in the document
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
    """
    Saves the BM25 scores to files, one per query.
    Args:
        scores (dict): The BM25 scores.
        output_folder (str): The directory to save output files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for query_id, doc_scores in scores.items():
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        output_file_path = os.path.join(output_folder, f"BM25_{query_id}Ranking.dat")
        with open(output_file_path, 'w') as file:
            for doc_id, score in sorted_docs:
                file.write(f"{doc_id} {score}\n")

# Example usage
query_file_path = 'C:/Users/pallavi/PycharmProjects/Assignment_Project2/the50Queries.txt'
base_data_directory = 'C:/Users/pallavi/PycharmProjects/Assignment_Project2/Data_Collection-1/Data_Collection'
output_folder = 'C:/Users/pallavi/PycharmProjects/Assignment_Project2/RankingOutputs11'

# Load and process queries and documents
queries = load_queries(query_file_path)
all_scores = {}

# Iterate through all data directories
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

# Save calculated scores
save_scores(all_scores, output_folder)

