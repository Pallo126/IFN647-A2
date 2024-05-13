import os

def load_all_relevance_judgments(base_path):
    all_relevance_judgments = {}
    for i in range(101, 151):  # Assuming files are named Dataset101.txt to Dataset150.txt
        file_path = os.path.join(base_path, f"Dataset{i}.txt")
        with open(file_path, 'r') as file:
            for line in file:
                query_id, doc_id, relevance = line.strip().split()
                if query_id not in all_relevance_judgments:
                    all_relevance_judgments[query_id] = {}
                all_relevance_judgments[query_id][doc_id] = int(relevance)
    return all_relevance_judgments

import math

def average_precision(retrieved_docs, relevant_docs):
    hit_count = 0
    cumulative_precision = 0.0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_docs:
            hit_count += 1
            cumulative_precision += hit_count / (i + 1)
    if hit_count == 0:
        return 0
    return cumulative_precision / hit_count

def precision_at_k(retrieved_docs, relevant_docs, k=10):
    hits = sum(1 for doc_id in retrieved_docs[:k] if doc_id in relevant_docs)
    return hits / k

def dcg_at_k(retrieved_docs, relevant_docs, k=10):
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevant_docs:
            dcg += 1 / math.log(i + 2, 2)  # log base 2 of (i+1)
    return dcg

def evaluate_model(all_scores, all_relevance_judgments):
    map_scores = []
    precision_at_10_scores = []
    dcg_at_10_scores = []

    for query_id, scores in all_scores.items():
        retrieved_docs = [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
        relevant_docs = set(doc_id for doc_id, rel in all_relevance_judgments.get(query_id, {}).items() if rel == 1)

        ap = average_precision(retrieved_docs, relevant_docs)
        p10 = precision_at_k(retrieved_docs, relevant_docs)
        dcg10 = dcg_at_k(retrieved_docs, relevant_docs)

        map_scores.append(ap)
        precision_at_10_scores.append(p10)
        dcg_at_10_scores.append(dcg10)

    mean_map = sum(map_scores) / len(map_scores) if map_scores else 0
    mean_p10 = sum(precision_at_10_scores) / len(precision_at_10_scores) if precision_at_10_scores else 0
    mean_dcg10 = sum(dcg_at_10_scores) / len(dcg_at_10_scores) if dcg_at_10_scores else 0

    return mean_map, mean_p10, mean_dcg10

# Load the relevance judgments
base_path = 'path_to_evaluation_benchmark_folder'
all_relevance_judgments = load_all_relevance_judgments(base_path)

# Assuming `all_scores` is loaded from your BM25 implementation
mean_map, mean_p10, mean_dcg10 = evaluate_model(all_scores, all_relevance_judgments)
print(f"Mean Average Precision: {mean_map}")
print(f"Precision at 10: {mean_p10}")
print(f"Mean DCG at 10: {mean_dcg10}")

