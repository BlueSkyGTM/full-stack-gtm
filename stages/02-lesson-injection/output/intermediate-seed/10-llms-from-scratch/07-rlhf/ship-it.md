## Ship It

**Tier 1 — Reward model only.** Train a reward model on 50 synthetic preference pairs using the Bradley-Terry objective. Split into 40 training / 10 held-out. Report training accuracy and held-out accuracy. If held-out accuracy is below 70%, increase preference pair diversity and retrain. This is the minimum viable quality classifier for any GTM pipeline that generates content at scale.

```python
import random
import torch
import torch.nn as nn
import torch.optim as optim

random.seed(42)
torch.manual_seed(42)

GOOD_TEMPLATES = [
    "the {topic} is {definition} because {reason}",
    "{topic} refers to {definition} and is used for {application}",
    "in brief {topic} means {definition} with {property}",
]
BAD_TEMPLATES = [
    "{topic} is a thing that some people know about",
    "{topic} has been around for a while and is {adjective}",
    "many experts discuss {topic} in their field of study",
]

TOPICS = ["git", "docker", "kubernetes", "terraform", "ansible", "jenkins", "grafana", "nginx"]
DEFINITIONS = ["a version control system", "a containerization platform", "an orchestration tool", "an IaC framework"]
REASONS = ["it tracks file changes", "it packages dependencies", "it scales services", "it automates deployment"]
APPLICATIONS = ["managing code history", "isolating environments", "coordinating clusters", "provisioning servers"]
PROPERTIES = ["stateless architecture", "declarative syntax", "event-driven design", "distributed execution"]
ADJECTIVES = ["important", "interesting", "relevant", "well-known"]

def generate_pair():
    topic = random.choice(TOPICS)
    good = random.choice(GOOD_TEMPLATES).format(
        topic=topic,
        definition=random.choice(DEFINITIONS),
        reason=random.choice(REASONS),
        application=random.choice(APPLICATIONS),
        property=random.choice(PROPERTIES),
    )
    bad = random.choice(BAD_TEMPLATES).format(
        topic=topic,
        adjective=random.choice(ADJECTIVES),
    )
    return good, bad

pairs = [generate_pair() for _ in range(50)]
train_pairs = pairs[:40]
test_pairs = pairs[40:]

def encode(text, vocab_size=200, max_len=15):
    tokens = [hash(w) % vocab_size for w in text.split()]
    tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)

class RewardNet(nn.Module):
    def __init__(self, vocab_size=200, embed_dim=32, hidden=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        e = self.embed(x).mean(dim=1)
        return self.fc2(torch.relu(self.fc1(e))).squeeze(-1)

model = RewardNet()
opt = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    total_loss = 0.0
    for good, bad in train_pairs:
        g = encode(good).unsqueeze(0)
        b = encode(bad).unsqueeze(0)
        gr = model(g)
        br = model(b)
        loss = -torch.log(torch.sigmoid(gr - br) + 1e-8).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
    if epoch % 100 == 0 or epoch == 299:
        print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_pairs):.4f}")

def evaluate(model, data):
    correct = 0
    for good, bad in data:
        with torch.no_grad():
            gr = model(encode(good).unsqueeze(0)).item()
            br = model(encode(bad).unsqueeze(0)).item()
        correct += int(gr > br)
    return correct / len(data)

train_acc = evaluate(model, train_pairs)
test_acc = evaluate(model, test_pairs)
print(f"\nTrain accuracy: {train_acc*100:.1f}%")
print(f"Test accuracy:  {test_acc*100:.1f}%")
```

**Tier 2 — Full PPO loop.** Implement a complete PPO training loop on a small policy (e.g., distilgpt2 or the toy policy above) with a pre-trained reward model. Log reward, KL divergence, clip fraction, and policy entropy at every step. Plot these metrics. You should see reward increase, KL stay bounded (if penalty is active), and clip fraction stabilize around 5-15%. If clip fraction goes to 0, your learning rate is too low. If it goes above 40%, your learning rate is too high.

**Tier 3 — Induce reward hacking.** Take the Tier 2 setup and set the KL penalty coefficient to zero. Run the same number of PPO steps. Document: (1) reward model scores — these will go *up* faster. (2) qualitative output quality — this will go *down*. Generate 20 samples before and after, and manually rate them. The gap between reward score and human quality rating is the reward hacking gap. This is the most important experiment in the lesson: it demonstrates why the KL penalty exists.