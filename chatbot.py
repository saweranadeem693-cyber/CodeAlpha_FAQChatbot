"""FAQ matching engine: NLTK preprocessing + TF-IDF + cosine similarity."""
import json
import re
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for pkg, path in [("punkt", "tokenizers/punkt"), ("punkt_tab", "tokenizers/punkt_tab"),
                  ("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

_LEMMA = WordNetLemmatizer()
_STOP = set(stopwords.words("english")) | {"please", "tell", "want", "know", "would", "like"}


# small synonym map so different wordings map to the same FAQ vocabulary
SYNONYMS = {"pay": "payment", "paying": "payment", "money": "refund", "abroad": "international",
            "overseas": "international", "parcel": "order", "package": "order", "open": "working",
            "hours": "hours", "timing": "hours", "timings": "hours", "buy": "order", "purchase": "order",
            "deliver": "delivery", "shipping": "delivery", "ship": "delivery", "coupon": "promo",
            "discount": "promo", "sale": "promo", "help": "support", "contact": "support",
            "call": "support", "email": "support", "login": "password", "sign": "password"}


def preprocess(text: str) -> str:
    """lowercase -> strip punctuation -> tokenize -> remove stopwords -> lemmatize"""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = [t for t in word_tokenize(text) if t not in _STOP]
    tokens = [_LEMMA.lemmatize(t) for t in tokens]
    return " ".join(SYNONYMS.get(t, t) for t in tokens)


class FAQChatbot:
    def __init__(self, faq_path: str = None, threshold: float = 0.15):
        faq_path = faq_path or Path(__file__).with_name("faqs.json")
        self.faqs = json.loads(Path(faq_path).read_text(encoding="utf-8"))
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(
            [preprocess(f["question"] + " " + f["question"] + " " + f["answer"]) for f in self.faqs])

    def ask(self, user_question: str):
        """Return (answer, matched_question, score)."""
        vec = self.vectorizer.transform([preprocess(user_question)])
        scores = cosine_similarity(vec, self.matrix)[0]
        best = int(scores.argmax())
        if scores[best] < self.threshold:
            return ("Sorry, I couldn't find an answer to that. Try rephrasing, or contact "
                    "support@example.com.", None, float(scores[best]))
        return self.faqs[best]["answer"], self.faqs[best]["question"], float(scores[best])


if __name__ == "__main__":  # simple CLI chat
    bot = FAQChatbot()
    print("FAQ Bot (type 'quit' to exit)")
    while (q := input("You: ").strip().lower()) not in {"quit", "exit"}:
        print("Bot:", bot.ask(q)[0])
