# Large Corpus Downloader

## Hook

You need 500k company records from a vendor API, a 40GB product review dataset from S3, and a week's worth of enriched leads from your enrichment pipeline. One connection drops halfway through. One API rate-limits you after 200 requests. One file arrives corrupted. You re-run everything from scratch. This beat introduces the class of problems that emerge when you move from "download a file" to "reliably ingest large datasets on a schedule."

---

## Concept

**Resumable downloads via HTTP Range requests.** The server advertises `Accept-Ranges: bytes`. The client sends `Range: bytes=1048576-` to resume from byte 1MB. If the server ignores the header, you get the full file — detect this by comparing `Content-Length` to your expected range size.

**Chunked parallel downloads.** Split a known file size into N byte ranges. Fetch each range concurrently. Write chunks to separate temp files, then concatenate. This is what `aria2c -x 16 -s 16` does under the hood — 16 connections, 16 segments per connection.

**Retry with exponential backoff and jitter.** On HTTP 429 or 5xx, wait `base_delay * 2^attempt + random_jitter`. Cap at a max delay. Track attempt count. This prevents thundering herd when a rate limit resets and all your workers retry simultaneously.

**Streaming writes.** For large files, `requests.get(url, stream=True)` fetches headers first, then you iterate `response.iter_content(chunk_size=8192)` and write to disk. The file never lives in memory in full. Without `stream=True`, a 40GB file will consume 40GB of RAM before the first byte hits disk.

**Integrity verification.** After download, compare the file's SHA-256 against a known hash. The server or manifest provides the expected hash. If mismatch, delete and retry. This catches silent corruption from interrupted writes or CDN errors.

**Directory-based locking.** Use `os.O_EXCL | os.O_CREAT` to atomically create a lock file. Write your PID inside. On crash, stale locks are detected by checking if the PID still exists. This prevents two download processes from writing the same file simultaneously.

---

## Demonstration

```python
import requests
import os
import hashlib
import time
import random

def download_with_resume(url, dest_path, max_retries=5):
    tmp_path = dest_path + ".tmp"
    downloaded = 0
    if os.path.exists(tmp_path):
        downloaded = os.path.getsize(tmp_path)
        print(f"Resuming from byte {downloaded}")

    headers = {"Range": f"bytes={downloaded}-"} if downloaded > 0 else {}
    attempt = 0

    while attempt < max_retries:
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=30)
            if resp.status_code == 429:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"Rate limited. Waiting {wait:.1f}s (attempt {attempt + 1})")
                time.sleep(wait)
                attempt += 1
                continue

            mode = "ab" if downloaded > 0 else "wb"
            with open(tmp_path, mode) as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            print(f"Download complete: {downloaded} bytes")
            os.rename(tmp_path, dest_path)
            return True

        except requests.exceptions.RequestException as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"Error: {e}. Retrying in {wait:.1f}s")
            time.sleep(wait)
            attempt += 1

    print(f"Failed after {max_retries} attempts")
    return False

def verify_sha256(file_path, expected_hash):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    match = actual == expected_hash
    print(f"SHA-256: {actual}")
    print(f"Expected: {expected_hash}")
    print(f"Match: {match}")
    return match

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
dest = "tinyshakespeare.txt"
download_with_resume(url, dest)

if os.path.exists(dest):
    size = os.path.getsize(dest)
    print(f"File size on disk: {size} bytes")
    with open(dest, "r") as f:
        first_line = f.readline().strip()
    print(f"First line: {first_line}")
```

Output confirms: resume detection, byte count on disk, first line of Shakespeare. Observable.

---

## Use It

**GTM redirect: foundational for Zone 1 (Data Foundation).**

Every enrichment pipeline, lead list sync, and vendor data feed you build in GTM depends on reliable file transfer. The pattern is the same whether you're downloading a 2GB Crunchbase snapshot, pulling weekly website traffic data from an S3 bucket your vendor provisioned, or syncing your Apollo export to a local data store before deduplication.

Specific scenario: your enrichment vendor provides a daily S3 drop of company data. You need to pull it, verify it, and stage it for your deduplication pipeline. If the download fails at 3am, the retry logic ensures it completes before your 6am enrichment run. If the file is corrupted, the SHA-256 check catches it and triggers an alert instead of feeding bad data into your Clay waterfall.

Exercise hooks:
- **Easy:** Download a public CSV of company data using the resumable downloader. Kill the process halfway, restart, confirm it resumes.
- **Medium:** Extend the function to log each attempt (timestamp, bytes transferred, status code) to a JSONL file for observability.
- **Hard:** Build a parallel chunk downloader that splits a large file across 4 concurrent connections using HTTP Range headers, reassembles the chunks, and verifies SHA-256.

---

## Ship It

Exercise hooks:
- **Easy:** Write a script that downloads a file, verifies its hash, and prints a single-line JSON summary: `{"file": "...", "bytes": ..., "sha256": "...", "status": "ok"}`.
- **Medium:** Wrap the downloader in a CLI tool that accepts a YAML manifest of multiple files (URL, destination, expected hash), downloads them all with resume support, and prints a summary table of successes and failures.
- **Hard:** Build a scheduled downloader that reads a manifest, tracks completed downloads in a SQLite database (URL, file path, hash, timestamp, status), and on re-run only downloads files that have changed (by comparing remote `ETag` or `Last-Modified` headers against stored values).

---

## Evaluate

Assessment hooks (not full questions):

- Given a scenario where `Content-Length` is 1GB but the downloader stops at 600MB with no error, diagnose the most likely cause and the header/flag that would prevent it.
- Given a log showing retry attempts at 1s, 2s, 4s, 8s, then immediate success, identify the backoff strategy and explain why jitter is added.
- Given a SHA-256 mismatch after a "successful" download, list three possible causes and the diagnostic step for each.
- Compare `stream=True` vs default behavior in `requests.get` for a 20GB file. Explain what happens to memory in each case.
- Given two processes downloading the same file to the same path simultaneously, explain the race condition and how `O_EXCL` lock files resolve it.