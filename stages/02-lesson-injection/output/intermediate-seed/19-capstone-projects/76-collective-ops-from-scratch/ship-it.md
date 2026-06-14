## Ship It

Build a `CollectiveOps` class that packages all primitives into a single testable unit. Every method prints input/output shapes and intermediate state. This is the class you will reach for when prototyping distributed workflows before committing to a real transport layer.

```python
class CollectiveOps:
    def __init__(self, n_workers):
        self.n_workers = n_workers
        self.workers = [[] for _ in range(n_workers)]
        self.root = 0
        self.log = []

    def _log(self, msg):
        self.log.append(msg)
        print(msg)

    def broadcast(self, data, root=0):
        n = self.n_workers
        self._log(f"\n[BROADCAST] root={root}, data_len={len(data)}, workers={n}")
        self._log(f"  input shape at root: [{len(data)}]")
        self._log(f"  input shape at others: []")
        outputs = []
        for i in range(n):
            copy = list(data)
            outputs.append(copy)
            self._log(f"  worker {i} output shape: [{len(copy)}]")
        self._log(f"  output: every worker has [{len(data)}]")
        self.workers = outputs
        return outputs

    def scatter(self, data, root=0):
        n = self.n_workers
        chunk_size = len(data) // n
        remainder = len(data) % n
        chunks = []
        idx = 0
        for i in range(n):
            size = chunk_size + (1 if i < remainder else 0)
            chunks.append(data[idx:idx + size])
            idx += size

        self._log(f"\n[SCATTER] root={root}, data_len={len(data)}, workers={n}")
        self._log(f"  input shape at root: [{len(data)}]")
        for i, chunk in enumerate(chunks):
            self._log(f"  worker {i} receives chunk[{len(chunk)}]: {chunk}")
        self.workers = chunks
        return chunks

    def gather(self, worker_chunks=None):
        if worker_chunks is None:
            worker_chunks = self.workers
        n = len(worker_chunks)
        self._log(f"\n[GATHER] workers={n}")
        gathered = []
        for i, chunk in enumerate(worker_chunks):
            self._log(f"  worker {i} sends chunk[{len(chunk)}]: {chunk}")
            gathered.extend(chunk)
        self._log(f"  root output shape: [{len(gathered)}]")
        self._log(f"  root output: {gathered}")
        return gathered

    def reduce(self, worker_data, op=sum):
        n = len(worker_data)
        self._log(f"\n[REDUCE] workers={n}, op={op.__name__}")
        for i, data in enumerate(worker_data):
            val = sum(data) if isinstance(data, list) else data
            self._log(f"  worker {i} contributes: {data} (local_sum={val})")
        total = sum(sum(d) if isinstance(d, list) else d for d in worker_data)
        self._log(f"  root output: {total}")
        return total

    def all_reduce(self, worker_data, op=sum):
        n = len(worker_data)
        self._log(f"\n[ALL-REDUCE] workers={n}, op={op.__name__}")
        self._log(f"  Step 1: gather all data to root")
        gathered = list(worker_data)
        self._log(f"  Step 2: reduce at root")
        total = sum(sum(d) if isinstance(d, list) else d for d in gathered)
        self._log(f"  reduced value: {total}")
        self._log(f"  Step 3: broadcast to all workers")
        result = [total] * n
        for i in range(n):
            self._log(f"  worker {i} output: {total}")
        return result

    def all_gather(self, worker_chunks=None):
        if worker_chunks is None:
            worker_chunks = self.workers
        n = len(worker_chunks)
        self._log(f"\n[ALL-GATHER] workers={n}")
        all_data = []
        for i, chunk in enumerate(worker_chunks):
            self._log(f"  worker {i} contributes: {chunk}")
            all_data.extend(chunk)
        self._log(f"  all workers receive: [{len(all_data)}] elements")
        outputs = [list(all_data) for _ in range(n)]
        self.workers = outputs
        return outputs

    def reduce_scatter(self, worker_partitioned_data, op=sum):
        n = len(worker_partitioned_data)
        n_parts = len(worker_partitioned_data[0])
        self._log(f"\n[REDUCE-SCATTER] workers={n}, partitions_per_worker={n_parts}")
        if n_parts != n:
            self._log(f"  WARNING: expected {n} partitions per worker, got {n_parts}")

        reduced_partitions = []
        for part_idx in range(n_parts):
            partition_values = []
            for w in range(n):
                part = worker_partitioned_data[w][part_idx]
                val = sum(part) if isinstance(part, list) else part
                partition_values.append(val)
            reduced = sum(partition_values)
            reduced_partitions.append(reduced)
            self._log(f"  partition {part_idx}: values={partition_values} -> reduced={reduced}")

        outputs = []
        for i in range(n):
            outputs.append(reduced_partitions[i])
            self._log(f"  worker {i} receives partition[{i}] = {reduced_partitions[i]}")

        self.workers = outputs
        return outputs


print("=" * 60)
print("TEST 1: Broadcast")
print("=" * 60)
ops = CollectiveOps(4)
ops.broadcast([1, 2, 3, 4])

print("\n" + "=" * 60)
print("TEST 2: Scatter")
print("=" * 60)
ops2 = CollectiveOps(3)
ops2.scatter([10, 20, 30, 40, 50, 60])

print("\n" + "=" * 60)
print("TEST 3: All-Reduce")
print("=" * 60)
ops3 = CollectiveOps(4)
result = ops3.all_reduce([[1, 2], [3, 4], [5, 6], [7, 8]])
print(f"\nAll-reduce result: {result}")

print("\n" + "=" * 60)
print("TEST 4: Reduce-Scatter")
print("=" * 60)
ops4 = CollectiveOps(4)
partitioned = [
    [[1, 1], [2, 2], [3, 3], [4, 4]],
    [[10, 10], [20, 20], [30, 30], [40, 40]],
    [[100, 100], [200, 200], [300, 300], [400, 400]],
    [[1000, 1000], [2000, 2000], [3000, 3000], [4000, 4000]],
]
rs_result = ops4.reduce_scatter(partitioned)
print(f"\nReduce-scatter result: {rs_result}")
print(f"Partition 0: {partitioned[0][0]} + {partitioned[1][0]} + {partitioned[2][0]} + {partitioned[3][0]} = {[sum(x) for x in [partitioned[0][0], partitioned[1][0], partitioned[2][0], partitioned[3][0]]]} -> {sum(sum(x) for x in [partitioned[0][0], partitioned[1][0], partitioned[2][0], partitioned[3][0]])}")
```

Output (abridged for key sections):

```
============================================================
TEST 1: Broadcast
============================================================

[BROADCAST] root=0, data_len=4, workers=4
  input shape at root: [4]
  input shape at others: []
  worker 0 output shape: [4]
  worker 1 output shape: [4]
  worker 2 output shape: [4]
  worker 3 output shape: [4]
  output: every worker has [4]

============================================================
TEST 4: Reduce-Scatter
============================================================

[REDUCE-SCATTER] workers=4, partitions_per_worker=4
  partition 0: values=[2, 20, 200, 2000] -> reduced=2222
  partition 1: values=[4, 40, 400, 4000] -> reduced=4444
  partition 2: values=[6, 60, 600, 6000] -> reduced=6666
  partition 3: values=[8, 80, 800, 8000] -> reduced=8888
  worker 0 receives partition[0] = 2222
  worker 1 receives partition[1] = 4444
  worker 2 receives partition[2] = 6666
  worker 3 receives partition[3] = 8888

Reduce-scatter result: [2222, 4444, 6666, 8888]
```

Each partition is reduced across all workers and the result lands at exactly one worker. Worker 0 gets the reduction of partition 0 from every worker. Worker 1 gets partition 1. No partition is duplicated, none are missing. This is the primitive that ZeRO and FSDP use to shard optimizer state and gradients.