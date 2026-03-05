"""
core/embeddings.py
Lightweight text embedding engine using TF-IDF with sublinear term frequency.
Falls back gracefully when sentence-transformers is unavailable.
Provides cosine similarity, MMR diversification, and BM25 scoring.
"""
from __future__ import annotations
import math
import re
import hashlib
from collections import Counter
from typing import Optional
import numpy as np


# ── Text preprocessing ──────────────────────────────────────────────────────
_STOP_WORDS = frozenset("""
a an the and or but in on at to for of with by from as is was are were
be been being have has had do does did will would could should may might
must shall can this that these those it its i me my we our you your he
she they them his her their what which who whom when where why how all
each both few more some such no not only same so than too very just
""".split())

_EMOTION_LEXICON = {
    # Positive emotions
    "happy": 1.0, "joy": 1.0, "love": 0.9, "excited": 0.9, "wonderful": 0.8,
    "great": 0.7, "good": 0.6, "nice": 0.6, "beautiful": 0.8, "amazing": 0.9,
    "fantastic": 0.9, "excellent": 0.8, "perfect": 0.9, "laugh": 0.7,
    "smile": 0.6, "celebrate": 0.8, "victory": 0.8, "win": 0.7, "success": 0.8,
    # Negative emotions
    "sad": -0.8, "angry": -0.9, "fear": -0.8, "hate": -0.9, "terrible": -0.9,
    "awful": -0.9, "horrible": -0.9, "bad": -0.6, "wrong": -0.5, "fail": -0.7,
    "death": -0.9, "die": -0.9, "kill": -1.0, "violence": -1.0, "fight": -0.7,
    "attack": -0.9, "war": -0.8, "suffer": -0.8, "pain": -0.7, "cry": -0.6,
    "destroy": -0.9, "danger": -0.8, "threat": -0.8, "disaster": -0.9,
    # Neutral/high intensity
    "intense": 0.3, "powerful": 0.4, "dramatic": 0.2, "tense": -0.2,
    "nervous": -0.3, "worried": -0.4, "confused": -0.2,
}

_BRAND_SAFETY_WORDS = {
    "violence": {"kill", "murder", "blood", "stab", "shoot", "gun", "weapon",
                 "bomb", "explosion", "attack", "war", "terrorist", "violence"},
    "adult": {"sex", "nude", "naked", "explicit", "adult", "porn", "erotic"},
    "hate": {"hate", "racist", "discrimination", "slur", "bigot", "extremist"},
    "illegal": {"drug", "cocaine", "heroin", "theft", "steal", "fraud", "crime",
                "illegal", "smuggle", "trafficking"},
    "profanity": {"damn", "hell", "crap"},  # keeping mild for demo
}

_IAB_CATEGORIES = {
    "IAB1": {"art", "music", "film", "movie", "song", "dance", "paint", "theater",
             "actor", "director", "entertainment", "show", "performance", "concert"},
    "IAB2": {"car", "auto", "vehicle", "drive", "race", "motor", "truck", "speed",
             "engine", "road", "travel", "transport"},
    "IAB3": {"business", "company", "corporate", "market", "finance", "money",
             "stock", "invest", "trade", "revenue", "profit", "startup"},
    "IAB4": {"career", "job", "work", "employment", "professional", "skill",
             "education", "training", "degree", "university", "school"},
    "IAB5": {"education", "learn", "study", "school", "teach", "knowledge",
             "science", "research", "academic", "student"},
    "IAB6": {"family", "home", "parent", "child", "baby", "marriage", "relationship",
             "house", "kitchen", "cooking", "recipe", "food", "meal"},
    "IAB7": {"health", "medical", "doctor", "hospital", "fitness", "exercise",
             "wellness", "diet", "nutrition", "medicine", "therapy"},
    "IAB8": {"food", "restaurant", "chef", "cooking", "recipe", "eat", "drink",
             "coffee", "wine", "beer", "cuisine", "taste"},
    "IAB9": {"game", "gaming", "sport", "team", "player", "match", "tournament",
             "competition", "fitness", "workout", "gym", "athlete"},
    "IAB10": {"home", "garden", "decor", "furniture", "interior", "design",
              "renovation", "diy", "craft", "house"},
    "IAB11": {"law", "legal", "government", "politics", "policy", "election",
              "democracy", "rights", "justice", "court"},
    "IAB12": {"news", "world", "international", "global", "country", "nation",
              "conflict", "crisis", "report", "journalist"},
    "IAB13": {"finance", "investment", "bank", "stock", "crypto", "money",
              "economy", "tax", "loan", "savings", "wealth"},
    "IAB14": {"society", "culture", "community", "social", "people", "human",
              "diversity", "equality", "change", "movement"},
    "IAB15": {"science", "technology", "research", "innovation", "space",
              "physics", "biology", "chemistry", "experiment", "discovery"},
    "IAB16": {"pet", "animal", "dog", "cat", "wildlife", "nature", "environment",
              "ocean", "forest", "bird", "zoo"},
    "IAB17": {"sport", "football", "basketball", "soccer", "tennis", "swimming",
              "olympics", "athlete", "team", "championship", "league"},
    "IAB18": {"fashion", "style", "clothing", "beauty", "makeup", "luxury",
              "brand", "model", "trend", "accessory"},
    "IAB19": {"tech", "software", "app", "computer", "internet", "digital",
              "ai", "robot", "code", "developer", "startup", "innovation"},
    "IAB20": {"travel", "destination", "hotel", "flight", "vacation", "tourism",
              "adventure", "explore", "country", "culture"},
    "IAB21": {"real estate", "property", "house", "apartment", "rent",
              "mortgage", "investment", "location"},
    "IAB22": {"shopping", "store", "buy", "product", "retail", "deal",
              "discount", "brand", "consumer"},
    "IAB23": {"religion", "faith", "spiritual", "church", "prayer",
              "belief", "meditation", "philosophy"},
}

_IAB_NAMES = {
    "IAB1": "Arts & Entertainment", "IAB2": "Automotive", "IAB3": "Business",
    "IAB4": "Career", "IAB5": "Education", "IAB6": "Family & Parenting",
    "IAB7": "Health & Fitness", "IAB8": "Food & Drink", "IAB9": "Hobbies & Interests",
    "IAB10": "Home & Garden", "IAB11": "Law & Government", "IAB12": "News",
    "IAB13": "Personal Finance", "IAB14": "Society", "IAB15": "Science",
    "IAB16": "Pets", "IAB17": "Sports", "IAB18": "Style & Fashion",
    "IAB19": "Technology & Computing", "IAB20": "Travel", "IAB21": "Real Estate",
    "IAB22": "Shopping", "IAB23": "Religion & Spirituality",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


def get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


# ── TF-IDF Vectorizer ────────────────────────────────────────────────────────
class TFIDFVectorizer:
    """Lightweight TF-IDF with sublinear TF scaling."""

    def __init__(self, max_features: int = 2000):
        self.max_features = max_features
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        self._fitted = False

    def fit(self, documents: list[str]) -> "TFIDFVectorizer":
        tokenized = [tokenize(d) for d in documents]
        n = len(tokenized)

        # Document frequency
        df: Counter = Counter()
        for tokens in tokenized:
            df.update(set(tokens))

        # Select top-N by df, filter very rare
        # Use min_df=1 for small corpora (≤20 docs), else min_df=2 for recall
        min_df = 1 if n <= 20 else 2
        selected = [term for term, freq in df.most_common(self.max_features)
                    if freq >= min_df]
        self.vocab = {term: i for i, term in enumerate(selected)}

        # IDF with smoothing
        self.idf = np.array([
            math.log((1 + n) / (1 + df[term])) + 1.0
            for term in selected
        ])
        self._fitted = True
        return self

    def transform(self, documents: list[str]) -> np.ndarray:
        """Return L2-normalised TF-IDF matrix (n_docs × vocab_size)."""
        V = len(self.vocab)
        mat = np.zeros((len(documents), V), dtype=np.float32)
        for i, doc in enumerate(documents):
            tokens = tokenize(doc)
            tf: Counter = Counter(tokens)
            total = max(len(tokens), 1)
            for term, cnt in tf.items():
                if term in self.vocab:
                    j = self.vocab[term]
                    # Sublinear TF
                    mat[i, j] = (1.0 + math.log(cnt)) * self.idf[j] / total
        # L2 normalise
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def fit_transform(self, documents: list[str]) -> np.ndarray:
        return self.fit(documents).transform(documents)

    def embed(self, text: str) -> np.ndarray:
        """Embed single document."""
        return self.transform([text])[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors."""
    return float(np.dot(a, b))


def cosine_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Batch cosine similarity: (n_queries × n_corpus)."""
    return queries @ corpus.T


# ── BM25 ─────────────────────────────────────────────────────────────────────
class BM25:
    """BM25 Okapi scoring."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens: list[list[str]] = []
        self.idf: dict[str, float] = {}
        self.avgdl: float = 0.0
        self.n: int = 0

    def fit(self, documents: list[str]) -> "BM25":
        self.corpus_tokens = [tokenize(d) for d in documents]
        self.n = len(self.corpus_tokens)
        self.avgdl = sum(len(t) for t in self.corpus_tokens) / max(self.n, 1)
        df: Counter = Counter()
        for tokens in self.corpus_tokens:
            df.update(set(tokens))
        self.idf = {
            term: math.log((self.n - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        return self

    def score(self, query: str, doc_idx: int) -> float:
        q_tokens = tokenize(query)
        doc = self.corpus_tokens[doc_idx]
        dl = len(doc)
        tf_map: Counter = Counter(doc)
        score = 0.0
        for qt in q_tokens:
            if qt not in self.idf:
                continue
            tf = tf_map.get(qt, 0)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1))
            score += self.idf[qt] * num / max(den, 1e-9)
        return score

    def score_all(self, query: str) -> np.ndarray:
        return np.array([self.score(query, i) for i in range(self.n)])


# ── Semantic analysis helpers ────────────────────────────────────────────────
def analyse_sentiment(text: str) -> dict:
    """Rule-based sentiment using emotion lexicon."""
    tokens = tokenize(text)
    scores = [_EMOTION_LEXICON.get(t, 0.0) for t in tokens]
    if not scores:
        return {"score": 0.0, "label": "neutral", "intensity": 0.0}
    mean = sum(scores) / len(scores)
    intensity = sum(abs(s) for s in scores) / len(scores)
    if mean > 0.2:
        label = "positive"
    elif mean < -0.2:
        label = "negative"
    else:
        label = "neutral"
    return {"score": round(mean, 3), "label": label, "intensity": round(intensity, 3)}


def extract_brand_safety(text: str) -> dict:
    """Return per-category flags and an overall safety score 0-1."""
    tokens = set(tokenize(text))
    flags: dict[str, bool] = {}
    for category, words in _BRAND_SAFETY_WORDS.items():
        flags[category] = bool(tokens & words)
    n_unsafe = sum(flags.values())
    safety_score = max(0.0, 1.0 - n_unsafe * 0.25)
    return {"flags": flags, "safety_score": round(safety_score, 3), "is_safe": n_unsafe == 0}


def classify_iab(text: str, top_n: int = 3) -> list[dict]:
    """Score text against IAB taxonomy and return top_n categories."""
    tokens = Counter(tokenize(text))
    scores: list[tuple[str, float]] = []
    for cat_id, keywords in _IAB_CATEGORIES.items():
        overlap = sum(tokens.get(kw, 0) for kw in keywords)
        if overlap > 0:
            score = overlap / max(len(tokens), 1)
            scores.append((cat_id, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [
        {"id": cat_id, "name": _IAB_NAMES.get(cat_id, cat_id), "score": round(sc, 4)}
        for cat_id, sc in scores[:top_n]
    ]


def extract_entities(text: str) -> dict:
    """
    Lightweight entity extraction using regex patterns and keyword matching.
    Returns products, locations, people (names), organisations.
    """
    # Product mentions (capitalised multi-word + known brand patterns)
    products = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}(?:\s+\d+)?\b", text)
    # Numbers with units → quantities/prices
    quantities = re.findall(r"\$[\d,]+\.?\d*|\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|%|kg|km|mph))", text, re.IGNORECASE)
    # All-caps abbreviations → organisations/tickers
    orgs = re.findall(r"\b[A-Z]{2,6}\b", text)
    # Filter overly common
    orgs = [o for o in orgs if o not in {"I", "II", "III", "OK", "TV", "US", "UK"}]
    return {
        "products": list(set(products))[:5],
        "quantities": list(set(quantities))[:5],
        "organisations": list(set(orgs))[:5],
    }


def extract_topics(text: str) -> list[str]:
    """Extract dominant topics using term frequency over IAB vocabulary."""
    tokens = Counter(tokenize(text))
    topic_scores: list[tuple[str, int]] = []
    for cat_id, keywords in _IAB_CATEGORIES.items():
        score = sum(tokens.get(kw, 0) for kw in keywords)
        if score > 0:
            topic_scores.append((_IAB_NAMES[cat_id], score))
    topic_scores.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in topic_scores[:4]]


def mmr_diversify(
    query_vec: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_ids: list,
    top_k: int = 10,
    lambda_param: float = 0.7,
) -> list:
    """
    Maximal Marginal Relevance diversification.
    Balances relevance to query (lambda) vs diversity from selected (1-lambda).
    """
    if len(candidate_ids) == 0:
        return []

    relevance = cosine_matrix(query_vec.reshape(1, -1), candidate_vecs)[0]
    selected = []
    remaining = list(range(len(candidate_ids)))

    while len(selected) < top_k and remaining:
        if not selected:
            best = max(remaining, key=lambda i: relevance[i])
        else:
            sel_vecs = candidate_vecs[selected]
            scores = []
            for i in remaining:
                rel = relevance[i]
                sim_to_sel = float(np.max(cosine_matrix(
                    candidate_vecs[i].reshape(1, -1), sel_vecs
                )))
                scores.append(lambda_param * rel - (1 - lambda_param) * sim_to_sel)
            best = remaining[int(np.argmax(scores))]

        selected.append(best)
        remaining.remove(best)

    return [candidate_ids[i] for i in selected]
