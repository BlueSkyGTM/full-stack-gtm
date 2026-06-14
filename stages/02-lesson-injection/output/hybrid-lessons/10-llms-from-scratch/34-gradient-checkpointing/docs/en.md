# Gradient Checkpointing and Activation Recomputation

## Learning Objectives
- Implement gradient checkpointing in a PyTorch training loop to reduce VRAM consumption.
- Compare the execution time and memory footprint differences between standard backpropagation and activation recomputation.
- Configure a Hugging Face transformer training script to prevent Out of Memory (OOM) errors on constrained hardware.

## The Problem

You are fine-tuning an 8-billion parameter language model to classify intent from raw, unstructured sales transcripts or to generate hyper-personalized cold emails based on technical documentation. You apply LoRA (Low-Rank Adaptation) to keep the weight memory low, loading the model in 4-bit precision. You initialize your batch size to 4 and your sequence length to 4,096 tokens to handle long conversational context. 

You launch the training script. The GPU memory spikes, the OS kills your process, and you receive a `CUDA out of memory` error. 

The weights fit in VRAM, but the training process crashes. Why? Because standard backpropagation requires storing all intermediate neuron outputs (activations) to calculate gradients later. For large sequence lengths, these intermediate activations scale linearly and consume massive amounts of VRAM. If you cannot fit the activations for a 4,096-token sequence into memory, you cannot train on long context windows, fundamentally limiting the quality of your GTM intent extraction model.

## The Concept

To understand the solution, you need to understand the mechanism of the PyTorch dynamic computation graph. 

During a standard forward pass (calculating the loss), PyTorch builds a graph of operations. For every matrix multiplication and layer normalization, PyTorch saves the intermediate outputs (activations) into VRAM. During the backward pass (calculating the gradients), PyTorch traverses this graph in reverse, using those saved activations to compute how much each weight contributed to the final error. The VRAM footprint scales linearly with the depth of the model, the batch size, and the sequence length.

**Gradient Checkpointing** (or Activation Recomputation) breaks this linear memory scaling by trading compute for memory. 

When checkpointing is enabled, PyTorch does not save the intermediate activations for every layer during the forward pass. It only saves the inputs to specific "checkpoint" boundaries. PyTorch drops the internal activations, freeing VRAM. 

During the backward pass, when PyTorch needs those intermediate activations to compute gradients, it takes the saved checkpoint input and simply runs the forward pass for that specific layer segment *again*. It recomputes the activation on the fly, uses it to calculate the gradient, and discards it again.

By discarding and recomputing, you drastically reduce peak VRAM usage, allowing you to fit larger sequence lengths into your GPU. The tradeoff is that the forward pass effectively runs twice, slowing down training throughput by roughly 20-30%—a highly favorable exchange if the alternative is a crash.

```mermaid
graph TD
    subgraph Standard Backpropagation
        A1[Forward Pass] -->|Saves all activations| A2[(High VRAM Usage)]
        A2 --> B1[Backward Pass]
        B1 -->|Uses saved activations| C1[Fast Gradients]
    end

    subgraph Gradient Checkpointing
        D1[Forward Pass] -->|Drops most activations| D2[(Low VRAM Usage)]
        D2 --> E1[Backward Pass]
        E1 -->|Re-runs Forward Pass locally| F1[Recomputed Activations]
        F1 -->|Calculates gradients| G1[Slower Throughput]
    end
```

## Build It

To see the memory and compute tradeoff in action, we will run a synthetic deep neural network in PyTorch. We will use a deep transformer block stack and measure both execution time and peak memory allocation under standard conditions versus gradient checkpointing.

Save this as `checkpoint_test.py` and run it.

```python
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import time

class DeepBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim * 4)
        self.linear_out = nn.Linear(dim * 4, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        x = torch.nn.functional.gelu(self.linear(x))
        return self.linear_out(x)

class DeepModel(nn.Module):
    def __init__(self, dim, depth, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.blocks = nn.ModuleList([DeepBlock(dim) for _ in range(depth)])

    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpointing:
                x = checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

batch_size = 16
seq_len = 1024
dim = 768
depth = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randn(batch_size, seq_len, dim, device=device)

model = DeepModel(dim, depth, use_checkpointing=False).to(device)
model.train()

start = time.time()
out = model(x)
loss = out.sum()
loss.backward()
standard_time = time.time() - start

if device == "cuda":
    torch.cuda.synchronize()
    standard_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    torch.cuda.reset_peak_memory_stats()
else:
    standard_mem = 0

model_ckpt = DeepModel(dim, depth, use_checkpointing=True).to(device)
model_ckpt.train()

start = time.time()
out = model_ckpt(x)
loss = out.sum()
loss.backward()
checkpoint_time = time.time() - start

if device == "cuda":
    torch.cuda.synchronize()
    checkpoint_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
else:
    checkpoint_mem = 0

print(f"Device: {device}")
print(f"Standard Backprop:")
print(f"  Time: {standard_time:.4f}s")
if device == "cuda":
    print(f"  Peak VRAM: {standard_mem:.2f} GB")

print(f"Gradient Checkpointing:")
print(f"  Time: {checkpoint_time:.4f}s")
if device == "cuda":
    print(f"  Peak VRAM: {checkpoint_mem:.2f} GB")
```

When you run this on a GPU, you will observe that the standard implementation runs faster but consumes significantly more VRAM. The checkpointed implementation reduces peak memory usage by discarding intermediate states, but the total execution time increases because PyTorch must recalculate those states during the backward pass. If run on a CPU, memory tracking is bypassed, but you will still clearly observe the execution time penalty incurred by recomputing the layers twice.

## Use It

This implements the mechanism of gradient checkpointing via Hugging Face's `TrainingArguments`, enabling the fine-tuning of large language models for outbound personalization (GTM Cluster 1.4) on hardware that would otherwise throw Out of Memory errors. 

By wrapping the transformer's internal layers to drop and recompute activations, we can safely double or triple the context window passed into the model. This is critical when processing raw prospect data, technical blogs, or long-form LinkedIn posts to generate highly personalized, multi-paragraph cold emails. The increased sequence length allows the model to ground its generation in more evidence, improving the semantic quality of the output.

```python
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import torch

model_id = "meta-llama/Meta-Llama-3-8B"
device_map = {"": 0}

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)

model.gradient_checkpointing_enable()

training_args = TrainingArguments(
    output_dir="./llama3-gtm-personalization",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_seq_length=4096,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    learning_rate=2e-5,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args
)

print("Training initialized with activation recomputation enabled.")
print("Targeting GTM Cluster 1.4: Personalization & Outbound Sequencing.")
```

In this configuration, passing a 4096-token sequence (about 3,000 words of prospect context) on a single 24GB consumer GPU is mathematically impossible without `gradient_checkpointing=True`. With it enabled, the VRAM footprint drops below the 24GB threshold, allowing the forward and backward passes to complete successfully. 

[CITATION NEEDED — concept: quantitative impact of sequence length on GTM personalization conversion rates]

## Exercises

### Easy
Modify the `Build It` script. Change the `batch_size` variable from `16` to `32` and the `seq_len` to `2048`. Run the script again. Observe how the VRAM requirements scale for the standard backpropagation method compared to the gradient checkpointing method.

### Medium
The `checkpoint.checkpoint` function allows for custom memory management. Modify the `DeepModel` class in the `Build It` script to use **selective checkpointing**—meaning it only applies checkpointing to every *other* layer in the loop (e.g., layers with an even index). 
*Hint: You can check `block_index % 2 == 0`.*
Run the script and compare the execution time and memory usage against full checkpointing. This demonstrates how engineers balance the compute-to-memory tradeoff.

### Hard
Investigate the `use_reentrant` parameter. Change the parameter from `False` to `True` in the `Build It` script. 
1. Run the script and observe what happens to the execution and memory.
2. Research the difference between reentrant and non-reentrant checkpoints in PyTorch. Write a brief explanation of why `use_reentrant=False` is the modern standard, particularly when working with custom autograd functions or mixed precision training (like `bfloat16`).

## Key Terms
- **Activations**: The intermediate tensor outputs of neural network layers during the forward pass, required to calculate gradients during the backward pass.
- **Backpropagation**: The algorithm used to calculate the gradient of the loss function with respect to the neural network's weights, traversing the network backward.
- **Gradient Checkpointing**: A memory optimization technique that discards intermediate activations during the forward pass and recomputes them on demand during the backward pass to save VRAM.
- **Out of Memory (OOM)**: A runtime error occurring when a process attempts to allocate more VRAM than is physically available on the GPU.
- **Sequence Length**: The number of tokens processed by the model at one time. Larger sequence lengths allow for larger context windows but scale VRAM requirements linearly.

## Sources
- PyTorch Documentation: *Gradient Checkpointing* (`torch.utils.checkpoint`)
- Hugging Face Documentation: *TrainingArguments* and Memory Management
- Chen, T. et al. (2016). *Training Deep Nets with Sublinear Memory Cost.* (The foundational paper outlining activation recomputation).