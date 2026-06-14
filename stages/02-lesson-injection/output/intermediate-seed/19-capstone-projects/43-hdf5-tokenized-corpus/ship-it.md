## Ship It

**Easy:** Write a script that tokenizes a directory of `.txt` files and writes them to a single HDF5 file with a document-boundary index. Print total tokens and per-document offsets.

```python
import numpy as np
import h5py
import os
import tempfile

temp_dir = tempfile.mkdtemp()
txt_dir = os.path.join(temp_dir, "docs")
os.makedirs(txt_dir)

sample_docs = {
    "product_overview.txt": "Our platform automates compliance reporting for fintech companies using ML-based document classification.",
    "pricing.txt": "Starter plan costs 500 per month. Pro plan costs 2000 per month with custom integrations included.",
    "faq.txt": "Question: Do you support SSO? Answer: Yes we support Okta Azure AD and Google Workspace identity providers.",
    "release_notes.txt": "Version 2.3 adds bulk export API webhook retry logic and improved dashboard loading performance.",
    "security.txt": "All data is encrypted at rest using AES 256 and in transit using TLS 1.3 we are SOC2 Type II certified.",
}

for fname, content in sample_docs.items():
    with open(os.path.join(txt_dir, fname), "w") as f:
        f.write(content)

vocab = {}
next_id = 1

def tokenize(text):
    global next_id
    ids = []
    for word in text.lower().split():
        clean = word.strip(".,;:!?\"'()[]{}:")
        if clean not in vocab:
            vocab[clean] = next_id
            next_id += 1
        ids.append(vocab[clean])
    return np.array(ids, dtype=np.uint32)

files = sorted(os.listdir(txt_dir))
all_tokens = []
boundaries = []
filenames = []
pos = 0

for fname in files:
    with open(os.path.join(txt_dir, fname)) as f:
        text = f.read()
    toks = tokenize(text)
    all_tokens.append(toks)
    boundaries.append((pos, pos + len(toks)))
    filenames.append(fname)
    pos += len(toks)

flat = np.concatenate(all_tokens)
bounds_arr = np.array(boundaries, dtype=np.int64)

output_path = os.path.join(temp_dir, "corpus_easy.h5")
with h5py.File(output_path, "w") as f:
    f.create_dataset("tokens", data=flat, chunks=(4096,))
    di = f.create_dataset("doc_index", data=bounds_arr)
    di.attrs["filenames"] = filenames

print(f"Output: {output_path}")
print(f"Total tokens: {len(flat)}")
print(f"Documents: {len(boundaries)}")
print(f"\nPer-document offsets:")
for fname, (start, end) in zip(filenames, boundaries):
    print(f"  {fname}: tokens[{start}:{end}] ({end - start} tokens)")
```

**Medium:** Add gzip compression at chunk size 8192, measure and print the size difference between compressed and uncompressed versions, and benchmark random-access read time for 10 random spans.

```python
import numpy as np
import h5py
import os
import tempfile
import time

temp_dir = tempfile.mkdtemp()
np.random.seed(42)

num_docs = 50
doc_lengths = np.random.randint(200, 800, size=num_docs)
total_tokens = doc_lengths.sum()
flat = np.random.randint(0, 50000, size=total_tokens).astype(np.uint32)

bounds = []
pos = 0
for length in doc_lengths:
    bounds.append((pos, pos + length))
    pos += length
bounds = np.array(bounds, dtype=np.int64)

raw_path = os.path.join(temp_dir, "corpus_raw.h5")
comp_path = os.path.join(temp_dir, "corpus_gzip.h5")

with h5py.File(raw_path, "w") as f:
    f.create_dataset("tokens", data=flat, chunks=(8192,))
    f.create_dataset("doc_index", data=bounds)

with h5py.File(comp_path, "w") as f:
    f.create_dataset(
        "tokens", data=flat, chunks=(8192,),
        compression="gzip", compression_opts=4,
    )
    f.create_dataset("doc_index", data=bounds)

raw_size = os.path.getsize(raw_path)
comp_size = os.path.getsize(comp_path)
raw_bytes = flat.nbytes

print(f"Token data (in-memory): {raw_bytes:,} bytes ({raw_bytes / 1024 / 1024:.2f} MB)")
print(f"HDF5 uncompressed:      {raw_size:,} bytes ({raw_size / 1024 / 1024:.2f} MB)")
print(f"HDF5 gzip-4:            {comp_size:,} bytes ({comp_size / 1024 / 1024:.2f} MB)")
print(f"Compression ratio:      {raw_size / comp_size:.2f}x")
print(f"Space saved:            {raw_size - comp_size:,} bytes ({(1 - comp_size/raw_size)*100:.1f}%)")

print("\nRandom-access read benchmark (10 spans):")
with h5py.File(comp_path, "r") as fc:
    tokens_c = fc["tokens"]
with h5py.File(raw_path, "r") as fr:
    tokens_r = fr["tokens"]

print(f"{'Span #':<8} {'Start':<10} {'Length':<8} {'Raw (µs)':<12} {'Gzip (µs)':<12} {'Match':<6}")
for i in range(10):
    start = np.random.randint(0, total_tokens - 512)
    length = np.random.randint(64, 512)
    t0 = time.perf_counter_ns()
    span_r = tokens_r[start : start + length]
    raw_us = (time.perf_counter_ns() - t0) / 1000

    t0 = time.perf_counter_ns()
    span_c = tokens_c[start : start + length]
    comp_us = (time.perf_counter_ns() - t0) / 1000

    match = np.array_equal(span_r, span_c)
    print(f"{i+1:<8} {start:<10} {length:<8} {raw_us:<12.1f} {comp_us:<12.1f} {str(match):<6}")
```

**Hard:** Implement a PyTorch `Dataset` subclass that reads from the HDF5 corpus, shuffles at the document-boundary level, and yields fixed-length token sequences with cross-document padding. Print a batch of token IDs and their source document indices.

```python
import numpy as np
import h5py
import os
import tempfile

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Install with: pip install torch")
    print("Running HDF5-only demo instead.\n")

temp_dir = tempfile.mkdtemp()
corpus_path = os.path.join(temp_dir, "corpus_hard.h5")

np.random.seed(123)
vocab_size = 1000
num_docs = 30
doc_lengths = np.random.randint(50, 300, size=num_docs)
total = doc_lengths.sum()

flat = np.random.randint(1, vocab_size, size=total).astype(np.uint32)
bounds = []
pos = 0
for length in doc_lengths:
    bounds.append((pos, pos + length))
    pos += length
bounds = np.array(bounds, dtype=np.int64)
doc_names = [f"doc_{i:04d}" for i in range(num_docs)]

with h5py.File(corpus_path, "w") as f:
    f.create_dataset("tokens", data=flat, chunks=(8192,), compression="gzip")
    di = f.create_dataset("doc_index", data=bounds)
    di.attrs["doc_names"] = doc_names

if HAS_TORCH:
    class HDF5CorpusDataset(Dataset):
        def __init__(self, h5_path, seq_len=64, pad_id=0):
            self.h5_path = h5_path
            self.seq_len = seq_len
            self.pad_id = pad_id
            self._file = None
            self._tokens = None
            self._doc_index = None
            self._doc_names = None

            with h5py.File(h5_path, "r") as f:
                self.total_tokens = f["tokens"].shape[0]
                self.doc_index = f["doc_index"][:]
                self.doc_names = list(f["doc_index"].attrs["doc_names"])

            self.doc_order = np.arange(len(self.doc_index))
            np.random.shuffle(self.doc_order)

            self.samples = []
            for doc_idx in self.doc_order:
                start, end = self.doc_index[doc_idx]
                doc_len = end - start
                num_seqs = max(1, (doc_len + seq_len - 1) // seq_len)
                for seq_idx in range(num_seqs):
                    seq_start = start + seq_idx * seq_len
                    seq_end = min(seq_start + seq_len, end)
                    actual_len = seq_end - seq_start
                    self.samples.append((doc_idx, seq_start, actual_len))

        def _ensure_open(self):
            if self._file is None:
                self._file = h5py.File(self.h5_path, "r")
                self._tokens = self._file["tokens"]

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            self._ensure_open()
            doc_idx, seq_start, actual_len = self.samples[idx]
            raw = self._tokens[seq_start : seq_start + actual_len]

            token_ids = np.full(self.seq_len, self.pad_id, dtype=np.int64)
            token_ids[:actual_len] = raw
            attention_mask = np.zeros(self.seq_len, dtype=np.int64)
            attention_mask[:actual_len] = 1

            return {
                "input_ids": torch.from_numpy(token_ids),
                "attention_mask": torch.from_numpy(attention_mask),
                "doc_idx": torch.tensor(doc_idx, dtype=torch.long),
                "actual_len": torch.tensor(actual_len, dtype=torch.long),
            }

    dataset = HDF5CorpusDataset(corpus_path, seq_len=64, pad_id=0)
    print(f"Corpus: {dataset.total_tokens} tokens, {len(dataset.doc_index)} documents")
    print(f"Dataset samples (doc-shuffled, seq_len=64): {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    batch = next(iter(dataloader))
    print(f"\nBatch keys: {list(batch.keys())}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"doc_idx: {batch['doc_idx'].tolist()}")
    print(f"actual_len: {batch['actual_len'].tolist()}")

    print(f"\nFirst sample in batch:")
    print(f"  Tokens (first 20): {batch['input_ids'][0][:20].tolist()}")
    print(f"  Mask (first 20):   {batch['attention_mask'][0][:20].tolist()}")
    doc_name = dataset.doc_names[batch['doc_idx'][0].item()]
    print(f"  Source doc: {doc_name}")

    unique_docs = sorted(set(dataset.samples[i][0] for i in range(len(dataset))))
    print(f"\nUnique documents in sample pool: {len(unique_docs)}")
    print(f"Documents accessed in first batch: {sorted(batch['doc_idx'].tolist())}")

else:
    print("Demonstrating document-level sampling without PyTorch:\n")
    seq_len = 64
    pad_id = 0

    doc_order = np.arange(len(bounds))
    np.random.shuffle(doc_order)

    samples = []
    for doc_idx in doc_order:
        start, end = bounds[doc_idx]
        doc_len = end - start
        num_seqs = max(1, (doc_len + seq_len - 1) // seq_len)
        for seq_idx in range(num_seqs):
            s = start + seq_idx * seq_len
            e = min(s + seq_len, end)
            samples.append((doc_idx, s, e - s))

    with h5py.File(corpus_path, "r") as f:
        tokens = f["tokens"]
        for i in range(3):
            doc_idx, seq_start, actual_len = samples[i]
            raw = tokens[seq_start : seq_start + actual_len]
            padded = np.full(seq_len, pad_id, dtype=np.uint32)
            padded[:actual_len] = raw
            print(f"Sample {i}: doc={doc_names[doc_idx]}, actual_len={actual_len}")
            print(f"  First 10 tokens: {padded[:10].tolist()}")
            print(f"  Padding: {seq_len - actual_len} tokens at id={pad_id}\n")
```