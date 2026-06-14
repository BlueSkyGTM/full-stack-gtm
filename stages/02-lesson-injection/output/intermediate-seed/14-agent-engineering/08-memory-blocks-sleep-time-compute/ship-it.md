## Ship It

Four production concerns dominate once you deploy a Letta-style agent with memory blocks and sleep-time compute.

**Core memory sizing.** Every character in a core memory block is in the prompt on every inference call. A 3,000-character block costs roughly 750 tokens per turn. With four blocks at