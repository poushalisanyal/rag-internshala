import json
import os

from retriever import Retriever

# ---------------------------------------
# Initialize Retriever
# ---------------------------------------

retriever = Retriever()

# ---------------------------------------
# Load Evaluation Questions
# ---------------------------------------

with open("evaluation/questions.json", "r") as f:
    test_cases = json.load(f)

# ---------------------------------------
# Metrics
# ---------------------------------------

hit = 0
recall = 0
mrr = 0

results = []

TOP_K = 3

# ---------------------------------------
# Evaluate
# ---------------------------------------

for sample in test_cases:

    question = sample["question"]
    expected_source = sample["source"]

    # Normalize expected source
    expected_source = os.path.basename(expected_source)

    # Retrieve documents
    docs, latency = retriever.retrieve(question, k=TOP_K)

    retrieved_sources = []

    rank = None

    for i, doc in enumerate(docs):

        source = doc.metadata.get("source")

        # Convert data\rag.md -> rag.md
        source_name = os.path.basename(source)

        retrieved_sources.append(source_name)

        if source_name == expected_source and rank is None:
            rank = i + 1

    # -------------------------
    # Hit Rate
    # -------------------------

    if expected_source in retrieved_sources:
        hit += 1

    # -------------------------
    # Recall@K
    # -------------------------

    if expected_source in retrieved_sources[:TOP_K]:
        recall += 1

    # -------------------------
    # Mean Reciprocal Rank
    # -------------------------

    if rank is not None:
        mrr += 1 / rank

    # -------------------------
    # Generate Answer
    # -------------------------

    answer = retriever.ask(question, k=TOP_K)

    results.append({

        "question": question,

        "expected_source": expected_source,

        "retrieved_sources": retrieved_sources,

        "answer": answer["answer"],

        "latency": latency,

        "token_usage": answer["token_usage"]

    })


# ---------------------------------------
# Final Scores
# ---------------------------------------

N = len(test_cases)

metrics = {

    "Total Questions": N,

    "Hit Rate": round(hit / N, 3),

    "Recall@3": round(recall / N, 3),

    "MRR": round(mrr / N, 3)

}


print("\nEvaluation Results\n")

for k, v in metrics.items():
    print(f"{k}: {v}")


# ---------------------------------------
# Save Results
# ---------------------------------------

output = {

    "metrics": metrics,

    "results": results

}


with open("evaluation/results.json", "w") as f:

    json.dump(output, f, indent=4)


print("\nResults saved to evaluation/results.json")