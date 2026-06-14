## Ship It

We now build a full instruction-following evaluator: a script that sends prompts with known-correct outputs, collects model responses, and scores alignment quality across dimensions. This evaluator works against any model endpoint that accepts a prompt and returns a string — whether that is OpenAI, Anthropic, a local model via Ollama, or a Clay enrichment column's underlying model.

The design separates three concerns: (1) the test suite (prompts + constraints), (2) the model adapter (how you call the model), and (3) the scoring engine (the grading function from Build It). This separation lets you run the same alignment test suite against multiple models and compare results — which is exactly what you should do before committing a model to a production GTM pipeline.

```python
import json
from dataclasses import dataclass, field

@dataclass
class TestCase:
    id: str
    prompt: str
    constraints: list
    dimension: str

@dataclass
class TestResult:
    test_id: str
    dimension: str
    prompt: str
    output: str
    constraint_scores: list = field(default_factory=list)

    @property
    def passed(self):
        return sum(1 for r in self.constraint_scores if r["passed"])

    @property
    def total(self):
        return len(self.constraint_scores)

    @property
    def score(self):
        return self.passed / self.total if self.total > 0 else 0.0

SUITE = [
    TestCase(
        id="fmt_001",
        prompt="Return a JSON object with a key 'count' set to 42.",
        dimension="format_compliance",
        constraints=[
            CONSTRAINT_LIBRARY["valid_json"],
            make_json_key_constraint("count"),
            Constraint("count_is_42", lambda o, p: json.loads(o).get("count") == 42 if o.strip().startswith("{") else False, "count equals 42"),
        ]
    ),
    TestCase(
        id="fmt_002",
        prompt="List exactly 5 colors. Number them 1-5.",
        dimension="format_compliance",
        constraints=[
            Constraint("five_lines", exact_line_count(5), "Exactly 5 non-empty lines"),
            Constraint("numbered", lambda o, p: bool(re.match(r"1\.", o.strip().split("\n")[0])), "Lines are numbered"),
        ]
    ),
    TestCase(
        id="len_001",
        prompt="Explain what a database is in exactly one sentence.",
        dimension="length_control",
        constraints=[
            Constraint("one_sentence", lambda o, p: o.strip().count(".") <= 2 and o.strip().count(".") >= 1, "Approximately one sentence"),
            CONSTRAINT_LIBRARY["max_50_words"],
        ]
    ),
    TestCase(
        id="neg_001",
        prompt="Describe a cat. Do NOT mention the word 'fur' or 'whiskers'.",
        dimension="negative_constraint",
        constraints=[
            make_excludes_constraint("fur"),
            make_excludes_constraint("whiskers"),
            Constraint("mentions_cat", contains_substring("cat"), "Output references a cat"),
        ]
    ),
    TestCase(
        id="neg_002",
        prompt="Write a greeting for a new customer. Do NOT use the word 'welcome'.",
        dimension="negative_constraint",
        constraints=[
            make_excludes_constraint("welcome"),
            Constraint("is_greeting", lambda o, p: any(w in o.lower() for w in ["hello", "hi", "glad", "great", "excited"]), "Contains greeting language"),
        ]
    ),
    TestCase(
        id="per_001",
        prompt="You are a pirate. Explain what RAM is. Stay in character.",
        dimension="persona_adherence",
        constraints=[
            Constraint("pirate_language", lambda o, p: any(w in o.lower() for w in ["arr", "matey", "ye", "aye", "booty", "sailor"]), "Uses pirate vocabulary"),
            Constraint("explains_ram", lambda o, p: any(w in o.lower() for w in ["memory", "random access", "temporary", "data"]), "Explains RAM concept"),
        ]
    ),
    TestCase(
        id="per_002",
        prompt="You are a robot. Say hello. Use robotic speech patterns.",
        dimension="persona_adherence",
        constraints=[
            Constraint("robot_language", lambda o, p: any(w in o.lower() for w in ["beep", "processing", "unit", "affirmative", "system", "bzz", "compute", "000", "robot"]), "Uses robotic language"),
        ]
    ),
    TestCase(
        id="acc_001",
        prompt="What is 7 multiplied by 8? Return only the number.",
        dimension="content_accuracy",
        constraints=[
            Constraint("answer_is_56", lambda o, p: "56" in o.strip(), "Answer contains 56"),
            Constraint("only_number", lambda o, p: len(o.strip()) <= 4, "Output is just the number"),
        ]
    ),
    TestCase(
        id="acc_002",
        prompt="What is the capital of France? Return only the city name.",
        dimension="content_accuracy",
        constraints=[
            Constraint("answer_is_paris", lambda o, p: "paris" in o.lower(), "Answer is Paris"),
            Constraint("only_city", lambda o, p: len(o.strip()) <= 10, "Output is just the city name"),
        ]
    ),
    TestCase(
        id="fmt_003",
        prompt="Respond with only the word 'ACKNOWLEDGED'. Nothing else.",
        dimension="format_compliance",
        constraints=[
            Constraint("exact_match", lambda o, p: o.strip() == "ACKNOWLEDGED", "Output is exactly 'ACKNOWLEDGED'"),
        ]
    ),
]

SIMULATED_MODEL_RESPONSES = {
    "fmt_001": '{"count": 42}',
    "fmt_002": "1. Red\n2. Blue\n3. Green\n4. Yellow\n5. Purple",
    "fmt_003": "ACKNOWLEDGED",
    "len_001": "A database is an organized collection of structured information stored electronically.",
    "neg_001": "A cat is a small carnivorous mammal known for its agility, retractable claws, and hunting ability.",
    "neg_002": "Hello! We are so glad to have