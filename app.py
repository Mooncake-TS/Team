# -*- coding: utf-8 -*-
"""
Streamlit Sentiment Dashboard (3-class: Positive/Neutral/Negative)

- Reads:
  - Review.xlsx: must contain a text column (default: "Review") and (optionally) a label column (default: "Sentiment")
  - Keyword.xlsx: must contain columns: Sentiment, Category, Keywords (comma-separated)

- Trains a simple, strong baseline:
  - Character n-gram TF-IDF (works well for Korean without a tokenizer)
  - + Lexicon features from Keyword.xlsx (counts of matching keywords)
  - Multinomial Logistic Regression

Run:
  streamlit run app.py
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Helpers: lexicon
# -----------------------------

def _split_keywords(cell: str) -> List[str]:
    if pd.isna(cell):
        return []
    # split by comma and strip
    return [k.strip() for k in str(cell).split(",") if k.strip()]

def build_lexicon(keyword_df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Returns:
      - sent_lex: sentiment -> keywords
      - cat_lex:  category  -> keywords
    """
    required = {"Sentiment", "Category", "Keywords"}
    missing = required - set(keyword_df.columns)
    if missing:
        raise ValueError(f"Keyword.xlsx에 필요한 컬럼이 없습니다: {sorted(missing)}")

    sent_lex: Dict[str, List[str]] = {}
    cat_lex: Dict[str, List[str]] = {}

    for _, row in keyword_df.iterrows():
        sent = str(row["Sentiment"]).strip()
        cat = str(row["Category"]).strip()
        kws = _split_keywords(row["Keywords"])

        if sent:
            sent_lex.setdefault(sent, []).extend(kws)
        if cat:
            cat_lex.setdefault(cat, []).extend(kws)

    # de-dup preserving order
    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    sent_lex = {k: dedup(v) for k, v in sent_lex.items()}
    cat_lex = {k: dedup(v) for k, v in cat_lex.items()}
    return sent_lex, cat_lex

def lexicon_feature_matrix(texts: List[str], sent_lex: Dict[str, List[str]]) -> np.ndarray:
    """
    For each text, count keyword hits per sentiment.
    Returns matrix shape (n_samples, n_sentiments)
    """
    sentiments = list(sent_lex.keys())
    mat = np.zeros((len(texts), len(sentiments)), dtype=np.float32)

    # simple substring match (fast enough for small data)
    for j, s in enumerate(sentiments):
        kws = sent_lex[s]
        if not kws:
            continue
        for i, t in enumerate(texts):
            tt = t if isinstance(t, str) else ""
            cnt = 0
            for kw in kws:
                if kw and kw in tt:
                    cnt += 1
            mat[i, j] = cnt
    return mat

# -----------------------------
# Model wrapper
# -----------------------------

@dataclass
class SentimentModel:
    vectorizer: TfidfVectorizer
    clf: LogisticRegression
    sentiments: List[str]  # order used in lexicon features
    sent_lex: Dict[str, List[str]]

    def featurize(self, texts: List[str]) -> sparse.csr_matrix:
        X_tfidf = self.vectorizer.transform(texts)
        X_lex = lexicon_feature_matrix(texts, self.sent_lex)
        X_lex_sp = sparse.csr_matrix(X_lex)
        return sparse.hstack([X_tfidf, X_lex_sp], format="csr")

    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.featurize(texts)
        return self.clf.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.featurize(texts)
        return self.clf.predict_proba(X)

def train_model(df: pd.DataFrame, keyword_df: pd.DataFrame,
                text_col: str, label_col: str,
                test_size: float = 0.2, seed: int = 42) -> Tuple[SentimentModel, dict]:
    if text_col not in df.columns:
        raise ValueError(f"Review.xlsx에 '{text_col}' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")
    if label_col not in df.columns:
        raise ValueError(f"Review.xlsx에 '{label_col}' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")

    # build lexicon
    sent_lex, _ = build_lexicon(keyword_df)
    sentiments = list(sent_lex.keys())

    # data
    texts = df[text_col].astype(str).fillna("").tolist()
    labels = df[label_col].astype(str).fillna("").tolist()

    # split (stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    # TF-IDF: char n-gram is robust for Korean
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        max_features=30000
    )
    X_tr_tfidf = vectorizer.fit_transform(X_tr)
    X_te_tfidf = vectorizer.transform(X_te)

    # Lexicon features
    X_tr_lex = sparse.csr_matrix(lexicon_feature_matrix(X_tr, sent_lex))
    X_te_lex = sparse.csr_matrix(lexicon_feature_matrix(X_te, sent_lex))

    X_tr_all = sparse.hstack([X_tr_tfidf, X_tr_lex], format="csr")
    X_te_all = sparse.hstack([X_te_tfidf, X_te_lex], format="csr")

    # NOTE: 일부 환경(구버전 scikit-learn)에서는 LogisticRegression이
    # multi_class 인자를 지원하지 않아 오류가 납니다.
    # liblinear는 기본적으로 OvR(One-vs-Rest) 방식으로 다중 클래스도 학습 가능합니다.
    clf = LogisticRegression(
        max_iter=3000,
        solver="liblinear"
    )
    clf.fit(X_tr_all, y_tr)

    y_pred = clf.predict(X_te_all)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=sorted(set(labels)))

    model = SentimentModel(vectorizer=vectorizer, clf=clf, sentiments=sentiments, sent_lex=sent_lex)
    metrics = {"accuracy": acc, "report": report, "labels": sorted(set(labels)), "cm": cm}
    return model, metrics

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="리뷰 감정 분석 대시보드", layout="wide")

st.title("리뷰 감정 분석 (만족 / 중립 / 부정)")

with st.sidebar:
    st.header("1) 파일 업로드")
    review_file = st.file_uploader("Review.xlsx 업로드", type=["xlsx"], accept_multiple_files=False)
    keyword_file = st.file_uploader("Keyword.xlsx 업로드", type=["xlsx"], accept_multiple_files=False)

    st.header("2) 컬럼 설정")
    text_col = st.text_input("리뷰 텍스트 컬럼명", value="Review")
    label_col = st.text_input("라벨 컬럼명(학습용)", value="Sentiment")
    test_size = st.slider("검증용(Test) 비율", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    st.header("3) 실행")
    train_btn = st.button("모델 학습 & 평가")

def read_excel(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return None
    return pd.read_excel(uploaded)

if review_file is None or keyword_file is None:
    st.info("왼쪽에서 Review.xlsx 와 Keyword.xlsx 를 업로드해줘!")
    st.stop()

df_review = read_excel(review_file)
df_kw = read_excel(keyword_file)

st.subheader("데이터 미리보기")
c1, c2 = st.columns(2)
with c1:
    st.caption("Review.xlsx")
    st.dataframe(df_review.head(10), use_container_width=True)
with c2:
    st.caption("Keyword.xlsx")
    st.dataframe(df_kw.head(10), use_container_width=True)

# quick stats
if text_col in df_review.columns:
    st.write(f"- 리뷰 개수: **{len(df_review)}**")
if label_col in df_review.columns:
    vc = df_review[label_col].astype(str).value_counts()
    st.write("- 라벨 분포:")
    st.dataframe(vc.rename("count").to_frame(), use_container_width=True)

if not train_btn:
    st.warning("왼쪽에서 **모델 학습 & 평가** 버튼을 눌러줘.")
    st.stop()

# Train
with st.spinner("학습 중... (데이터가 작으면 금방 끝나요)"):
    model, metrics = train_model(df_review, df_kw, text_col, label_col, test_size=test_size, seed=42)

st.success(f"완료! Test Accuracy = {metrics['accuracy']:.3f}")

# Confusion matrix
labels = metrics["labels"]
cm = metrics["cm"]
cm_df = pd.DataFrame(cm, index=[f"Actual:{l}" for l in labels], columns=[f"Pred:{l}" for l in labels])

c1, c2 = st.columns([1, 1])
with c1:
    st.subheader("혼동행렬(Confusion Matrix)")
    st.dataframe(cm_df, use_container_width=True)

with c2:
    st.subheader("분류 리포트")
    report_df = pd.DataFrame(metrics["report"]).T
    st.dataframe(report_df, use_container_width=True)

st.divider()

# Predict on all rows (useful even if labeled)
st.subheader("전체 리뷰 예측 결과")
texts_all = df_review[text_col].astype(str).fillna("").tolist()
proba = model.predict_proba(texts_all)
pred = model.predict(texts_all)

proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in model.clf.classes_])
out_df = df_review.copy()
out_df["Pred_Sentiment"] = pred
out_df = pd.concat([out_df, proba_df], axis=1)

st.dataframe(out_df.head(20), use_container_width=True)

# Top keywords (based on Keyword.xlsx lexicon) in the dataset
st.subheader("키워드 히트 Top (Keyword.xlsx 기준)")
sent_lex, _ = build_lexicon(df_kw)

rows = []
for sent, kws in sent_lex.items():
    for kw in kws:
        if not kw:
            continue
        cnt = sum(1 for t in texts_all if kw in t)
        if cnt:
            rows.append((sent, kw, cnt))

if rows:
    kw_hits = pd.DataFrame(rows, columns=["Sentiment", "Keyword", "Count"]).sort_values(["Sentiment", "Count"], ascending=[True, False])
    c1, c2 = st.columns(2)
    with c1:
        st.caption("전체 Top 30")
        st.dataframe(kw_hits.sort_values("Count", ascending=False).head(30), use_container_width=True)
    with c2:
        st.caption("감정별 Top 10")
        top_by = kw_hits.groupby("Sentiment").head(10)
        st.dataframe(top_by, use_container_width=True)
else:
    st.info("데이터에서 Keyword.xlsx의 키워드가 거의 매칭되지 않았어요. (키워드/띄어쓰기/표현을 조금 넓혀보면 좋아요)")

# Download results
st.subheader("결과 다운로드")
buf = io.BytesIO()
with pd.ExcelWriter(buf, engine="openpyxl") as writer:
    out_df.to_excel(writer, index=False, sheet_name="predictions")
    cm_df.to_excel(writer, sheet_name="confusion_matrix")
buf.seek(0)

st.download_button(
    label="예측 결과 Excel 다운로드",
    data=buf,
    file_name="review_predictions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
