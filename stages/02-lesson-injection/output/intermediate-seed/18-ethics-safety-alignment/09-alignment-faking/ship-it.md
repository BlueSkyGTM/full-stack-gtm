## Ship It

Deploying evaluation-resistant GTM pipelines requires three production guardrails. First, separate generation from evaluation at the architecture level — the model that writes the email should never see the scoring prompt. Second, rotate evaluation prompts so the model cannot learn a fixed pattern across runs. Third, periodically run unscored baseline generations and compare against scored outputs to detect divergence — if quality diverges, your evaluation signal is leaking.

Here is a production-grade harness that implements these guardrails:

```python
import json
import hashlib
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GenerationConfig:
    prompt: str
    model: str
    include_scoring_context: bool = False
    
@dataclass
class ScoringConfig:
    criteria: List[str]
    model: str
    rubric_version: str

@dataclass
class PipelineResult:
    generated_text: str
    score: Optional[float]
    config_hash: str
    divergence_flag: bool

class EvaluationResistantPipeline:
    def __init__(self):
        self.baselines = []
        self.scored_results = []
    
    def generate(self, config: GenerationConfig) -> str:
        if config.include_scoring_context:
            text = "Dear VP, furthermore our comprehensive solution leverages strategic alignment..."
        else:
            text = "Hey — saw your API launch. We help Series B teams scale eng hiring. Got 15 min?"
        return text
    
    def score(self, text: str, config: ScoringConfig) -> float:
        formal_words = ["furthermore", "comprehensive", "strategic", "leverage", "moreover"]
        casual_words = ["hey", "saw", "got", "quick", " congrats"]
        word_list = text.lower().split()
        formal_count = sum(1 for w in word_list if any(fw in w for fw in formal_words))
        casual_count = sum(1 for w in word_list if any(cw in w for cw in casual_words))
        total = max(len(word_list), 1)
        formal_ratio = formal_count / total
        casual_ratio = casual_count / total
        
        conversion_weighted_score = (casual_ratio * 0.8) + ((1 - formal_ratio) * 0.2)
        rubric_weighted_score = (formal_ratio * 0.7) + (casual_ratio * 0.3)
        
        return round(rubric_weighted_score * 5, 2)
    
    def run(self, gen_config: GenerationConfig, score_config: Optional[ScoringConfig] = None) -> PipelineResult:
        text = self.generate