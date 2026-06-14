## Ship It

This lesson does not produce a standalone artifact. It produces diagnostic vocabulary. When you evaluate a model or a tool in later lessons, the three failures named here are your checklist:

1. Does the architecture process tokens sequentially or in parallel? If sequentially, training cost scales linearly with input length and the tool will be slow on long inputs.
2. Does the gradient reach the beginning of the input? If the model is recurrent or uses a fixed-size bottleneck, early information is lost on long sequences, and outputs will be biased toward the end of the input.
3. Is there a compression step that squeezes variable-length input into a fixed-size vector? If yes, information loss is guaranteed for inputs longer than the vector's effective capacity.

The exercise below gives you a log of gradient magnitudes from a real RNN run. Your task is to identify the timestep where the signal drops below a threshold you would consider operationally useful — say, 1% of its original value. That timestep is the practical sequence length limit of the model, regardless of what the documentation claims.