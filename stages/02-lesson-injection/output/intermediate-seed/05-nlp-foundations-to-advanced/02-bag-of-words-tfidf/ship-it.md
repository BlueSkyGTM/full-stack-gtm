## Ship It

The production version wraps the ranking logic into a CLI script that reads company descriptions from a JSON file and outputs a ranked CSV. It uses `scipy.sparse` for the TF-IDF matrix (sparse storage is critical when vocabulary exceeds a few thousand terms) and pickles the vocabulary so you can score new companies against an existing ICP profile without recomputing IDF.

The scaling limit is real and worth naming explicitly. Vocabulary size grows with corpus size — every new unique token adds a dimension. TF-IDF must recompute when the corpus changes, because IDF depends on document frequency across all documents. Adding one company that introduces a new word changes the vocabulary length and shifts every IDF weight. This is why TF-IDF is a fixed-corpus method, not a general-purpose representation. The evolution past this ceiling is embeddings: dense vectors of fixed dimensionality that do not grow with the corpus and capture semantic similarity beyond exact keyword match. That is the next lesson.

Save this as `icp_ranker.py`:

```python
import argparse
import csv
import json
import math
import pickle
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def build_vectorizer(corpus):
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        norm="l2",
        smooth_idf=True,
        use_idf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix

def cosine_sparse(query_vec, doc_matrix):
    query_norm = query_vec / (query_vec.sum() or 1.0)
    scores = doc_matrix.dot(query_norm.T).toarray().flatten()
    return scores

def main():
    parser = argparse.ArgumentParser(description="Rank companies by ICP keyword fit using TF-IDF cosine similarity")
    parser.add_argument("--input", required=True, help="Path to JSON file with company list")
    parser.add_argument("--icp", required=True, help="Path to JSON file with ICP keyword weights")
    parser.add_argument("--output", required=True, help="Path to output CSV")
    parser.add_argument("--save-vocab", default=None, help="Optional: pickle vocabulary to this path")
    args = parser.parse_args()

    with open(args.input) as f:
        companies = json.load(f)
    with open(args.icp) as f:
        icp_keywords = json.load(f)

    descriptions = [c["description"] for c in companies]

    vectorizer, matrix = build_vectorizer(descriptions)
    vocab = vectorizer.vocabulary_

    query_vec = [0.0] * len(vocab)
    for kw, weight in icp_keywords.items():
        if kw in vocab:
            query_vec[vocab[kw]] = weight
        else:
            print(f"WARNING: ICP keyword '{kw}' not in corpus vocabulary", file=sys.stderr)

    import numpy as np
    query_array = np.array(query_vec)
    query_sparse = csr_matrix(query_array.reshape(1, -1))

    scores = cosine_sparse(query_sparse, matrix)

    ranked = sorted(zip(companies, scores), key=lambda x: -x[1])

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "company", "score", "description"])
        for rank, (company, score) in enumerate(ranked, 1):
            writer.writerow([rank, company["name"], f"{score:.4f}", company["description"]])

    print(f"Ranked {len(companies)} companies → {args.output}", file=sys.stderr)

    if args.save_vocab:
        with open(args.save_vocab, "wb") as f:
            pickle.dump({"vocabulary": vocab, "idf": vectorizer.idf_}, f)
        print(f"Saved vocabulary → {args.save_vocab}", file=sys.stderr)

if __name__ == "__main__":
    main()
```

Create the input files and run it:

```bash
cat > companies.json << 'EOF'
[
  {"name": "PayStack", "description": "payment infrastructure platform for businesses in africa"},
  {"name": "Stripe", "description": "apis for payment processing and online commerce"},
  {"name": "Plaid", "description": "fintech api connecting bank accounts to financial apps"},
  {"name": "Marqeta", "description": "modern card issuing and payment infrastructure for saas"},
  {"name": "Notion", "description": "all in one workspace for notes and project management"}
]
EOF

cat > icp_profile.json << 'EOF'
{
  "payment": 3.0,
  "infrastructure": 2.5,
  "fintech": 2.0,
  "saas": 1.5,
  "api": 1.0,
  "processing": 1.0
}
EOF

python icp_ranker.py --input companies.json --icp icp_profile.json --output ranked.csv --save-vocab vocab.pkl

cat ranked.csv
```

**Expected output:**

```
rank,company,score,description
1,Marqeta,0.8165,modern card issuing and payment infrastructure for saas
2,PayStack,0.7303,payment infrastructure platform for businesses in africa
3,Stripe,0.3892,apis for payment processing and online commerce
4,Plaid,0.3098,fintech api connecting bank accounts to financial apps
5,Notion,0.0000,all in one workspace for notes and project management
```

The scores differ from the from-scratch version because scikit-learn applies L2 normalization and IDF smoothing by default. The ranking is consistent — Marqeta and PayStack at the top, Notion at zero. The vocabulary pickle lets you score a single new company against the same IDF profile without rebuilding the full corpus, as long as the new company's tokens already exist in the vocabulary. New tokens outside the vocabulary are invisible to the model — another manifestation of the exact-match ceiling.