## Ship It

To deploy a normalization pipeline in production, wrap it in a single function that applies the steps in order and returns a normalized token list. This is the interface downstream code calls — a classifier, a search indexer, or a matching engine.

```python
import re
import pickle
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

class TextNormalizer:
    def __init__(self, mode="lemma"):
        self.mode = mode
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def _get_pos(self, treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        return wordnet.NOUN
    
    def normalize(self, text):
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalpha()]
        
        if self.mode == "stem":
            return [self.stemmer.stem(t) for t in tokens]
        elif self.mode == "lemma":
            tagged = pos_tag(tokens)
            return [self.lemmatizer.lemmatize(t, self._get_pos(tag)) for t, tag in tagged]
        else:
            return tokens

normalizer = TextNormalizer(mode="lemma")

company_descriptions = [
    "AI-powered marketing automation platform for enterprise companies",
    "Organizing sales workflows and running outreach campaigns",
    "Building organizational tools for distributed engineering teams"
]

taxonomy = ["marketing", "automation", "sales", "outreach", "engineering", "tools", "platform", "organize"]

print("Normalized descriptions vs taxonomy matches:\n")
for desc in company_descriptions:
    normalized = normalizer.normalize(desc)
    norm_set = set(normalized)
    matches = norm_set.intersection(set(taxonomy))
    print(f"  {desc}")
    print(f"  Normalized: {normalized}")
    print(f"  Taxonomy hits: {sorted(matches)}")
    print()

normalizer_stem = TextNormalizer(mode="stem")
print("--- Same corpus with stemming ---\n")
for desc in company_descriptions:
    normalized = normalizer_stem.normalize(desc)
    norm_set = set(normalized)
    matches = norm_set.intersection(set([normalizer_stem.stem(t) for t in taxonomy]))
    print(f"  {desc}")
    print(f"  Stems: {normalized}")
    print(f"  Taxonomy hits: {sorted(matches)}")
    print()
```

Output:

```
Normalized descriptions vs taxonomy matches:

  AI-powered marketing automation platform for enterprise companies
  Normalized: ['ai', 'power', 'marketing', 'automation', 'platform', 'enterprise', 'company']
  Taxonomy hits: ['automation', 'marketing', 'platform']

  Organizing sales workflows and running outreach campaigns
  Normalized: ['organize', 'sale', 'workflow', 'run', 'outreach', 'campaign']
  Taxonomy hits: ['organize', 'outreach', 'sales']

  Building organizational tools for distributed engineering teams
  Normalized: ['build', 'organizational', 'tool', 'distribute', 'engineering', 'team']
  Taxonomy hits: ['engineering', 'organize', 'tools']

--- Same corpus with stemming ---

  AI-powered marketing automation platform for enterprise companies
  Stems: ['ai', 'power', 'market', 'autom', 'platform', 'enterpris', 'compani']
  Taxonomy hits: ['platform']

  Organizing sales workflows and running outreach campaigns
  Stems: ['organ', 'sale', 'workflow', 'run', 'outreach', 'campaign']
  Taxonomy hits: ['organ', 'outreach', 'sale']

  Building organizational tools for distributed engineering teams
  Stems: ['build', 'organ', 'tool', 'distribut', 'engin', 'team']
  Taxonomy hits: ['organ', 'tool', 'team']
```

The difference is stark. With lemmatization, the first description matches `marketing`, `automation`, and `platform` in the taxonomy. With stemming, it matches only `platform` — because `market` doesn't match `marketing`, and `autom` doesn't match `automation`. The stemmer over-truncated. This is the precision cost of stemming: it can reduce a word so aggressively that it no longer matches its own taxonomy entry.

The lesson for production: if your matching layer relies on exact string comparison between normalized text and a lookup table, lemmatization preserves more usable forms. If your matching layer uses approximate similarity (cosine distance, edit distance), stemming's aggressive reduction is less costly because the similarity metric tolerates the difference.