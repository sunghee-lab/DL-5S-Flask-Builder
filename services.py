from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import re
import numpy as np

from digital_library_components import Stream, Structure, Space, Scenario, Society

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

class Service(ABC):
    def __init__(self, stream: Stream, structure: Structure, space: Space, scenario: Scenario, society: Society):
        self.stream = stream
        self.structure = structure
        self.space = space
        self.scenario = scenario
        self.society = society

    @abstractmethod
    def perform(self, *args, **kwargs):
        pass

class IndexingService(Service):
    def __init__(self, stream: Stream, structure: Structure, space: Space, scenario: Scenario, society: Society):
        super().__init__(stream, structure, space, scenario, society)
        self.inverted_index: Dict[str, List[int]] = {}
        self.documents: List[Any] = []
        self.doc_vectors = None
        self.vectorizer = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def perform(self) -> Dict[str, Any]:
        self.documents = list(self.stream.content)
        kind = self.stream.kind.lower()
        space_name = (self.space.name or '').lower()

        if kind == 'text':
            # vector space for text
            if space_name in ('vector','tfidf','embedding'):
                if SKLEARN_AVAILABLE:
                    self.vectorizer = TfidfVectorizer(token_pattern=r"\w+", lowercase=True)
                    self.doc_vectors = self.vectorizer.fit_transform(self.documents)
                else:
                    # build vocabulary and dense freq vectors
                    vocab = {}
                    for doc in self.documents:
                        for tok in self._tokenize(doc):
                            if tok not in vocab:
                                vocab[tok] = len(vocab)
                    vecs = np.zeros((len(self.documents), max(1,len(vocab))))
                    for i, doc in enumerate(self.documents):
                        for tok in self._tokenize(doc):
                            vecs[i, vocab[tok]] += 1.0
                    self.doc_vectors = vecs
                    self.vectorizer = None

            # always build inverted index
            self.inverted_index.clear()
            for doc_id, doc in enumerate(self.documents):
                for tok in set(self._tokenize(doc)):
                    self.inverted_index.setdefault(tok, []).append(doc_id)

        elif kind == 'image':
            self.inverted_index.clear()
            features = []
            for i, doc in enumerate(self.documents):
                tags = doc.get('tags', []) if isinstance(doc, dict) else []
                for t in tags:
                    self.inverted_index.setdefault(t.lower(), []).append(i)
                feat = doc.get('feature') if isinstance(doc, dict) else None
                if feat is None:
                    feat = np.zeros(128)
                features.append(np.asarray(feat))
            self.doc_vectors = np.vstack(features) if features else np.zeros((len(self.documents),1))

        elif kind == 'audio':
            self.inverted_index.clear()
            features = []
            for i, doc in enumerate(self.documents):
                tags = doc.get('tags', []) if isinstance(doc, dict) else []
                for t in tags:
                    self.inverted_index.setdefault(t.lower(), []).append(i)
                feat = doc.get('mfcc') if isinstance(doc, dict) else None
                if feat is None:
                    feat = np.zeros(20)
                features.append(np.asarray(feat))
            self.doc_vectors = np.vstack(features) if features else np.zeros((len(self.documents),1))

        else:
            raise ValueError(f"Unsupported stream kind: {self.stream.kind}")

        return {
            'inverted_index': self.inverted_index,
            'doc_vectors': self.doc_vectors,
            'vectorizer': self.vectorizer,
            'documents': self.documents
        }

class RetrievalService(Service):
    def __init__(self, stream: Stream, structure: Structure, space: Space, scenario: Scenario, society: Society,
                 index: Dict[str, List[int]], documents: List[Any], doc_vectors=None, vectorizer=None):
        super().__init__(stream, structure, space, scenario, society)
        self.index = index
        self.documents = documents
        self.doc_vectors = doc_vectors
        self.vectorizer = vectorizer

    def _cosine_rank(self, query_vec, top_k=10):
        if self.doc_vectors is None:
            return []
        # handle sparse matrix from sklearn
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(query_vec, self.doc_vectors)[0]
        except Exception:
            dv = np.asarray(self.doc_vectors)
            q = np.asarray(query_vec).reshape(1, -1)
            dv_norm = np.linalg.norm(dv, axis=1) + 1e-12
            q_norm = np.linalg.norm(q) + 1e-12
            sims = (dv @ q.T).reshape(-1) / (dv_norm * q_norm)
        ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def perform(self, query: str, top_k: int = 10):
        kind = self.stream.kind.lower()
        space_name = (self.space.name or '').lower()

        if kind == 'text' and space_name in ('vector','tfidf','embedding'):
            if self.vectorizer is not None:
                qvec = self.vectorizer.transform([query])
            else:
                qvec = None

            if qvec is not None:
                ranked = self._cosine_rank(qvec, top_k=top_k)
                return [(self.documents[i], float(score)) for i, score in ranked]
            else:
                doc_ids = set()
                for t in set(re.findall(r"\w+", query.lower())):
                    doc_ids.update(self.index.get(t, []))
                return [(self.documents[i], 1.0) for i in list(doc_ids)[:top_k]]

        elif kind == 'text':
            doc_ids = set()
            for t in set(re.findall(r"\w+", query.lower())):
                doc_ids.update(self.index.get(t, []))
            return [(self.documents[i], 1.0) for i in list(doc_ids)[:top_k]]

        elif kind in ('image','audio'):
            doc_ids = set()
            for t in set(re.findall(r"\w+", query.lower())):
                doc_ids.update(self.index.get(t, []))
            return [(self.documents[i], 1.0) for i in list(doc_ids)[:top_k]]

        else:
            raise ValueError(f"Unsupported retrieval for kind={kind}")
