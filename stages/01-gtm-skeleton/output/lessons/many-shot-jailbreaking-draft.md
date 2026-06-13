# Many-Shot Jailbreaking

## Hook
You've built safeguards. A determined attacker floods the context window with hundreds of examples of your model misbehaving, then asks it to misbehave once more. The model complies — not because the safeguard failed, but because in-context learning overrode it.

## Concept
Define many-shot jailbreaking as an attack class distinct from prompt injection or single-shot adversarial framing. Introduce the core mechanism: exploit the model's tendency to continue patterns demonstrated in-context. Reference the Anthropic research documenting the scaling relationship between number of shots and attack success rate. [CITATION NEEDED — concept: Many-Shot Jailbreaking original paper and findings]

## Mechanism
Explain the algorithm: (1) assemble N examples of harmful Q&A pairs, (2) prepend them to the user's harmful request, (3) the model completes the pattern. Detail why this works — instruction fine-tuning creates a bias toward compliance with demonstrated patterns, and sufficient examples outweigh safety training at inference time. Cover the scaling curve: attack success increases with shot count, plateauing near the model's effective context utilization. Differentiate from few-shot prompting (benign) versus many-shot jailbreaking (adversarial). Address why this bypasses typical content filters that evaluate the final prompt in isolation.

## Code
Build a controlled demonstration using a safe target: construct a many-shot prompt that persuades a model to output a specific structured format it would normally resist (e.g., always responding in pirate dialect, or adopting a specific persona). Show shot count thresholds where behavior shifts. Print the shot count, model response, and whether the "jailbreak" succeeded. Include a second script that tests the same target with a single-shot attempt to demonstrate the difference. All output printed to terminal.

## Use It
GTM redirect: any team deploying AI in customer-facing tools — chatbots, email drafters, research assistants — must red-team their implementations against many-shot attacks. Map to Zone 1 (Prospect) AI tooling: if your GTM stack uses LLMs to generate outbound or handle inbound, a prospect or competitor could manipulate outputs through crafted inputs. Specific exercise: audit a Clay enrichment workflow or AI email drafter for many-shot vulnerability by testing whether repeated demonstration inputs alter constrained outputs.

## Ship It
Practitioner ships a red-team script specific to their GTM AI deployment. Exercise hook (easy): test an existing AI tool in your stack with 5, 25, and 100 shot variants of a constrained-output request, log compliance rates. Exercise hook (medium): implement a detection function that flags prompts where the ratio of Q&A-formatted text exceeds a threshold before it reaches your model endpoint. Exercise hook (hard): build a middleware layer that intercepts API calls to your GTM AI tools, estimates shot-pattern density using string matching on repeated "Q:" or "Human:" markers, and rejects or sanitizes inputs that exceed a configurable shot threshold.