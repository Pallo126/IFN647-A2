import os
import math
import re
from collections import defaultdict, Counter

# Constants
lambda_ = 0.4

# Paths
data_dir = 'C:/Users/samin/Desktop/IFN647/Assignment 2/Data_Collection-1/Data_Collection/'
query_file_path = 'C:/Users/samin/Desktop/IFN647/Assignment 2/the50Queries.txt'
output_dir = 'C:/Users/samin/Desktop/IFN647/Assignment 2/RankingOutputs2/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def tokenize(text):
    """ Simple tokenizer to split and lower case the text. """
    return re.findall(r'\w+', text.lower())

def load_data(directory):
    """ Loads and tokenizes documents from each subdirectory within the directory. """
    docs = defaultdict(list)
    lengths = defaultdict(int)
    collection_freq = Counter()
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        print(f"Processing folder: {folder_name}")
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = tokenize(file.read())
                docs[folder_name].append((filename, content))
                lengths[folder_name] += len(content)
                collection_freq.update(content)
        print(f"Loaded {len(docs[folder_name])} documents from {folder_name}")
    return docs, lengths, collection_freq

def load_queries(filename):
    """ Parses queries from the given file. """
    queries = {}
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
        print("Reading queries from file...")
        for match in re.finditer(r'<top>(.*?)</top>', content, re.DOTALL):
            num = re.search(r'<num> Number: R(\d+)', match.group(1)).group(1)
            title = re.search(r'<title> (.*)', match.group(1)).group(1)
            queries[num] = tokenize(title)
        print(f"Loaded {len(queries)} queries.")
    return queries

def compute_jm_lm_scores(docs, doc_lengths, collection_freq, total_length, queries):
    """ Computes the Jelinek-Mercer Language Model scores. """
    results = defaultdict(list)
    print("Computing scores...")
    for query_num, query_terms in queries.items():
        dataset = f'Data_C{int(query_num) + 100}'
        for filename, content in docs[dataset]:
            score = 0
            doc_length = len(content)
            doc_freq = Counter(content)
            for term in query_terms:
                doc_prob = doc_freq[term] / doc_length if doc_length else 0
                collection_prob = collection_freq[term] / total_length if total_length else 0
                term_score = (1 - lambda_) * doc_prob + lambda_ * collection_prob
                score += math.log(term_score) if term_score > 0 else 0
            results[query_num].append((filename, score))
    print("Score computation complete.")
    return results

def save_results(results, output_dir):
    """ Saves computed scores to the output directory. """
    print("Saving results...")
    for query_num, scores in results.items():
        output_file_path = os.path.join(output_dir, f'JM_LM_R{query_num}Ranking.dat')
        scores.sort(key=lambda x: x[1], reverse=True)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            for filename, score in scores:
                file.write(f'{filename}\t{score}\n')
        print(f"Results for query {query_num} saved to {output_file_path}")

# Load data
docs, doc_lengths, collection_freq = load_data(data_dir)
total_length = sum(doc_lengths.values())
queries = load_queries(query_file_path)

# Compute scores
results = compute_jm_lm_scores(docs, doc_lengths, collection_freq, total_length, queries)

# Save results
save_results(results, output_dir)
print("Ranking complete.")
