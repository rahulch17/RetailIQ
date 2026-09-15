from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = Path("knowledge_base")

class Retriever:
    def __init__(self):
        self.docs = []

        for p in KB.glob("*"):
            if p.is_file():
                self.docs.append(
                    (p.name, p.read_text(encoding="utf-8"))
                )

        self.vectorizer = TfidfVectorizer(stop_words="english")

        self.matrix = (
            self.vectorizer.fit_transform([d[1] for d in self.docs])
            if self.docs else None
        )

    def search(self, query, k=3):
        if self.matrix is None:
            return []

        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        idx = scores.argsort()[::-1][:k]

        return [
            {
                "document": self.docs[i][0],
                "score": float(scores[i]),
                "text": self.docs[i][1],
            }
            for i in idx
            if scores[i] > 0
        ]
