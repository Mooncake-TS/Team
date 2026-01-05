# -*- coding: utf-8 -*-
"""
Streamlit Sentiment Dashboard (3-class: Positive/Neutral/Negative)

- Auto-loads Review.xlsx / Keyword.xlsx if they exist next to this app.py (repo files)
- Upload is optional (if uploaded, uploaded files take priority)
- Trains baseline:
  - Character n-gram TF-IDF (tokenizer 없이 한국어에 강함)
  - + Lexicon features from Keyword.xlsx (키워드 매칭 카운트)
  - Logistic Regression (liblinear: 구버전 sklearn 호환)

Run:
  streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

def _split_keywords(cell) -> List[str]:
    if pd.isna(cell):
        return []
    return [k.strip() for k in str(cell).split(",") if k.strip()]

def build_lexicon(keyword_df: pd.DataFrame) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Returns:
      - sent_lex: sentiment -> keywords
      - cat_lex:  category  -> keywords
    Required columns: Sentiment, Category, Keywords
    """
    required = {"Sentiment", "Category", "Keywords"}
    missing = required - set(keyword_df.columns)
    if missing:
        raise ValueError(
            f"Keyword.xlsx에 필요한 컬럼이 없습니다: {sorted(missing)}\n"
            f"현재 컬럼: {keyword_df.columns.tolist()}"
        )

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
    sentiments: List[str]          # order used in lexicon features
    sent_lex: Dict[str, List[str]] # sentiment -> keywords

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

def train_model(
    df: pd.DataFrame,
    keyword_df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = 0.2,
    seed: int = 42
) -> Tuple[SentimentModel, dict]:

    if text_col not in df.columns:
        raise ValueError(f"Review.xlsx에 '{text_col}' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")
    if label_col not in df.columns:
        raise ValueError(f"Review.xlsx에 '{label_col}' 컬럼이 없습니다. 현재 컬럼: {df.columns.tolist()}")

    # labels 정리(공백/NaN 제거)
    work = df[[text_col, label_col]].copy()
    work[text_col] = work[text_col].astype(str).fillna("").str.strip()
    work[label_col] = work[label_col].astype(str).fillna("").str.strip()
    work = work[(work[text_col] != "") & (work[label_col] != "")]
    if len(work) < 10:
        raise ValueError("학습 가능한 행이 너무 적어요. (리뷰/라벨 빈칸이 많을 수 있음)")

    # build lexicon
    sent_lex, _ = build_lexicon(keyword_df)
    sentiments = list(sent_lex.keys())

    texts = work[text_col].tolist()
    labels = work[label_col].tolist()

    # 클래스가 2개 미만이면 학습 불가
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError(f"라벨 종류가 1개뿐이라 학습이 안 됩니다. 현재 라벨: {unique_labels}")

    # split (가능하면 stratify)
    stratify = labels if len(unique_labels) > 1 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=stratify
    )

    # TF-IDF: char n-gram
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

    clf = LogisticRegression(
        max_iter=3000,
        solver="liblinear"  # 구버전 sklearn 호환
    )
    clf.fit(X_tr_all, y_tr)

    y_pred = clf.predict(X_te_all)
    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=unique_labels)

    model = SentimentModel(vectorizer=vectorizer, clf=clf, sentiments=sentiments, sent_lex=sent_lex)
    metrics = {"accuracy": acc, "report": report, "labels": unique_labels, "cm": cm}
    return model, metrics

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="리뷰 감정 분석 대시보드", layout="wide")
st.title("리뷰 감정 분석 (만족 / 중립 / 부정)")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_REVIEW = APP_DIR / "Review.xlsx"
DEFAULT_KEYWORD = APP_DIR / "Keyword.xlsx"

@st.cache_data(show_spinner=False)
def load_excel_from_path(p: Path) -> pd.DataFrame:
    return pd.read_excel(p)

@st.cache_data(show_spinner=False)
def load_excel_from_upload(uploaded) -> pd.DataFrame:
    return pd.read_excel(uploaded)

with st.sidebar:
    st.header("1) 파일")
    st.caption("✅ 레포에 Review.xlsx / Keyword.xlsx가 있으면 자동으로 읽어요.\n(업로드하면 업로드 파일이 우선입니다)")
    review_file = st.file_uploader("Review.xlsx 업로드(선택)", type=["xlsx"], accept_multiple_files=False)
    keyword_file = st.file_uploader("Keyword.xlsx 업로드(선택)", type=["xlsx"], accept_multiple_files=False)

    st.header("2) 컬럼 설정")
    text_col = st.text_input("리뷰 텍스트 컬럼명", value="Review")
    label_col = st.text_input("라벨 컬럼명(학습용)", value="Sentiment")
    test_size = st.slider("검증(Test) 비율", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    st.header("3) 실행")
    train_btn = st.button("모델 학습 & 평가")

# ---- Load files (upload > local repo) ----
df_review: Optional[pd.DataFrame] = None
df_kw: Optional[pd.DataFrame] = None

load_msgs = []

try:
    if review_file is not None:
        df_review = load_excel_from_upload(review_file)
        load_msgs.append("Review.xlsx: 업로드 파일 사용")
    elif DEFAULT_REVIEW.exists():
        df_review = load_excel_from_path(DEFAULT_REVIEW)
        load_msgs.append(f"Review.xlsx: 레포 파일 사용 ({DEFAULT_REVIEW.name})")

    if keyword_file is not None:
        df_kw = load_excel_from_upload(keyword_file)
        load_msgs.append("Keyword.xlsx: 업로드 파일 사용")
    elif DEFAULT_KEYWORD.exists():
        df_kw = load_excel_from_path(DEFAULT_KEYWORD)
        load_msgs.append(f"Keyword.xlsx: 레포 파일 사용 ({DEFAULT_KEYWORD.name})")
except Exception as e:
    st.error(f"엑셀 로딩 중 오류: {e}")
    st.stop()

if df_review is None or df_kw is None:
    st.warning(
        "Review.xlsx / Keyword.xlsx를 찾지 못했어요.\n\n"
        "✅ 해결 방법:\n"
        "- 레포(프로젝트) 루트에 **Review.xlsx, Keyword.xlsx**를 올리기\n"
        "- 또는 왼쪽 사이드바에서 업로드하기"
    )
    st.stop()

st.success(" / ".join(load_msgs))

# ---- Preview ----
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
    vc = df_review[label_col].astype(str).fillna("").str.strip()
    vc = vc[vc != ""].value_counts()
    st.write("- 라벨 분포(빈칸 제외):")
    st.dataframe(vc.rename("count").to_frame(), use_container_width=True)
else:
    st.info(f"라벨 컬럼 '{label_col}' 이(가) 없으면 학습/평가가 불가해요. (예측만 하려면 학습된 모델이 필요)")

if not train_btn:
    st.warning("왼쪽에서 **모델 학습 & 평가** 버튼을 눌러야 예측 기능이 활성화돼요.")
    st.stop()

# ---- Train ----
try:
    with st.spinner("학습 중..."):
        model, metrics = train_model(df_review, df_kw, text_col, label_col, test_size=test_size, seed=42)
except Exception as e:
    st.error(f"학습 실패: {e}")
    st.stop()

st.success(f"완료! Test Accuracy = {metrics['accuracy']:.3f}")

# Confusion matrix + report (그래프 없이 표로만)
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

# ---- Predict all rows ----
st.subheader("전체 리뷰 예측 결과(상위 일부)")
texts_all = df_review[text_col].astype(str).fillna("").tolist()
proba = model.predict_proba(texts_all)
pred = model.predict(texts_all)

proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in model.clf.classes_])
out_df = df_review.copy()
out_df["Pred_Sentiment"] = pred
out_df = pd.concat([out_df, proba_df], axis=1)

st.dataframe(out_df.head(30), use_container_width=True)

st.divider()

# ---- One-line input predict ----
st.subheader("리뷰 한 줄 입력 → 감정 예측")
user_text = st.text_area("리뷰를 입력하세요", placeholder="예) 기사님이 너무 친절하고 시간도 정확했어요!", height=120)
do_pred = st.button("예측하기")

if do_pred:
    txt = (user_text or "").strip()
    if not txt:
        st.warning("리뷰 내용을 입력해줘!")
    else:
        pred_label = model.predict([txt])[0]
        proba1 = model.predict_proba([txt])[0]
        classes = list(model.clf.classes_)
        proba_table = pd.DataFrame({"class": classes, "prob": proba1}).sort_values("prob", ascending=False)

        st.write(f"### ✅ 예측 결과: **{pred_label}**")
        st.dataframe(proba_table, use_container_width=True)

st.divider()

# ---- Keyword hit table (그래프 없이 표만) ----
st.subheader("키워드 히트 Top (표로만 표시)")
try:
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
        kw_hits = (
            pd.DataFrame(rows, columns=["Sentiment", "Keyword", "Count"])
            .sort_values(["Sentiment", "Count"], ascending=[True, False])
        )
        c1, c2 = st.columns(2)
        with c1:
            st.caption("전체 Top 30")
            st.dataframe(kw_hits.sort_values("Count", ascending=False).head(30), use_container_width=True)
        with c2:
            st.caption("감정별 Top 10")
            st.dataframe(kw_hits.groupby("Sentiment").head(10), use_container_width=True)
    else:
        st.info("데이터에서 Keyword.xlsx 키워드 매칭이 거의 없어요. (표현/띄어쓰기/키워드 범위를 조금 넓히면 좋아요)")
except Exception as e:
    st.warning(f"키워드 분석은 건너뛰었어요: {e}")

# ---- Download ----
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
