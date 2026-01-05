# -*- coding: utf-8 -*-
"""
Reuse utilities outside Streamlit.

Example:

from sentiment_model import train_from_excel, predict_from_excel

model, metrics = train_from_excel("Review.xlsx", "Keyword.xlsx")
pred_df = predict_from_excel(model, "Review.xlsx")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

def _split_keywords(cell: str) -> List[str]:
    if pd.isna(cell):
        return []
    return [k.strip() for k in str(cell).split(",") if k.strip()]

def build_lexicon(keyword_df: pd.DataFrame) -> Dict[str, List[str]]:
    required = {"Sentiment", "Keywords"}
    missing = required - set(keyword_df.columns)
    if missing:
        raise ValueError(f"Keyword.xlsx에 필요한 컬럼이 없습니다: {sorted(missing)}")

    sent_lex: Dict[str, List[str]] = {}
    for _, row in keyword_df.iterrows():
        sent = str(row["Sentiment"]).strip()
        kws = _split_keywords(row["Keywords"])
        if sent:
            sent_lex.setdefault(sent, []).extend(kws)

    # de-dup
    for k in list(sent_lex.keys()):
        seen = set()
        out = []
        for w in sent_lex[k]:
            if w not in seen:
                out.append(w)
                seen.add(w)
        sent_lex[k] = out
    return sent_lex

def lexicon_feature_matrix(texts: List[str], sent_lex: Dict[str, List[str]]) -> np.ndarray:
    sentiments = list(sent_lex.keys())
    mat = np.zeros((len(texts), len(sentiments)), dtype=np.float32)
    for j, s in enumerate(sentiments):
        kws = sent_lex[s]
        for i, t in enumerate(texts):
            tt = t if isinstance(t, str) else ""
            mat[i, j] = sum(1 for kw in kws if kw and kw in tt)
    return mat

@dataclass
class SentimentModel:
    vectorizer: TfidfVectorizer
    clf: LogisticRegression
    sent_lex: Dict[str, List[str]]

    def featurize(self, texts: List[str]) -> sparse.csr_matrix:
        X_tfidf = self.vectorizer.transform(texts)
        X_lex = sparse.csr_matrix(lexicon_feature_matrix(texts, self.sent_lex))
        return sparse.hstack([X_tfidf, X_lex], format="csr")

    def predict(self, texts: List[str]) -> np.ndarray:
        return self.clf.predict(self.featurize(texts))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        return self.clf.predict_proba(self.featurize(texts))

def train_from_df(df: pd.DataFrame, kw_df: pd.DataFrame,
                  text_col: str="Review", label_col: str="Sentiment",
                  test_size: float=0.2, seed: int=42) -> Tuple[SentimentModel, dict]:
    if text_col not in df.columns:
        raise ValueError(f"Review.xlsx에 '{text_col}' 컬럼이 없습니다.")
    if label_col not in df.columns:
        raise ValueError(f"Review.xlsx에 '{label_col}' 컬럼이 없습니다.")

    texts = df[text_col].astype(str).fillna("").tolist()
    labels = df[label_col].astype(str).fillna("").tolist()

    sent_lex = build_lexicon(kw_df)

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,5), max_features=30000)
    X_tr_t = vectorizer.fit_transform(X_tr)
    X_te_t = vectorizer.transform(X_te)

    X_tr_l = sparse.csr_matrix(lexicon_feature_matrix(X_tr, sent_lex))
    X_te_l = sparse.csr_matrix(lexicon_feature_matrix(X_te, sent_lex))

    X_tr_all = sparse.hstack([X_tr_t, X_tr_l], format="csr")
    X_te_all = sparse.hstack([X_te_t, X_te_l], format="csr")

    clf = LogisticRegression(max_iter=3000, multi_class="auto")
    clf.fit(X_tr_all, y_tr)

    y_pred = clf.predict(X_te_all)

    labels_sorted = sorted(set(labels))
    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "report": classification_report(y_te, y_pred, output_dict=True, zero_division=0),
        "cm": confusion_matrix(y_te, y_pred, labels=labels_sorted),
        "labels": labels_sorted
    }
    return SentimentModel(vectorizer, clf, sent_lex), metrics

def train_from_excel(review_xlsx: str, keyword_xlsx: str,
                     text_col: str="Review", label_col: str="Sentiment",
                     test_size: float=0.2, seed: int=42):
    df = pd.read_excel(review_xlsx)
    kw = pd.read_excel(keyword_xlsx)
    return train_from_df(df, kw, text_col=text_col, label_col=label_col, test_size=test_size, seed=seed)

def predict_from_excel(model: SentimentModel, review_xlsx: str, text_col: str="Review") -> pd.DataFrame:
    df = pd.read_excel(review_xlsx)
    texts = df[text_col].astype(str).fillna("").tolist()
    proba = model.predict_proba(texts)
    pred = model.predict(texts)

    proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in model.clf.classes_])
    out = df.copy()
    out["Pred_Sentiment"] = pred
    return pd.concat([out, proba_df], axis=1)
