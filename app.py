# app.py
from __future__ import annotations

import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# -------------------------
# Utils
# -------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    return str(s).strip()

def build_lexicon(df_kw: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Keyword.xlsx 를 바탕으로 감정별 키워드 사전 생성.
    - 기대 컬럼: Sentiment / Keyword (또는 유사)
    """
    cols = [c.lower() for c in df_kw.columns]
    # best-effort 컬럼 찾기
    def find_col(candidates):
        for cand in candidates:
            for i, c in enumerate(cols):
                if cand == c:
                    return df_kw.columns[i]
        return None

    sent_col = find_col(["sentiment", "label", "감정", "라벨"])
    kw_col = find_col(["keyword", "키워드", "word", "단어"])

    if sent_col is None or kw_col is None:
        raise ValueError(
            "❌ Keyword.xlsx 컬럼을 찾지 못했어요. "
            "예: Sentiment / Keyword 컬럼(또는 감정/키워드) 형태로 맞춰주세요."
        )

    lex: Dict[str, List[str]] = {}
    sentiments: List[str] = []

    for _, r in df_kw.iterrows():
        s = normalize_text(r[sent_col])
        k = normalize_text(r[kw_col])
        if not s or not k:
            continue
        if s not in lex:
            lex[s] = []
            sentiments.append(s)
        lex[s].append(k)

    # 중복 제거
    for s in lex:
        lex[s] = sorted(list(set(lex[s])), key=len, reverse=True)

    return lex, sentiments

def train_model(
    df_review: pd.DataFrame,
    df_kw: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    3-class 감정 분류: 만족/중립/부정 (라벨 문자열 그대로 사용)
    """
    if text_col not in df_review.columns:
        raise ValueError(f"❌ Review.xlsx에 텍스트 컬럼이 없어요: {text_col}")
    if label_col not in df_review.columns:
        raise ValueError(f"❌ Review.xlsx에 라벨 컬럼이 없어요: {label_col}")

    # 결측/공백 제거
    df = df_review[[text_col, label_col]].copy()
    df[text_col] = df[text_col].astype(str).fillna("").map(normalize_text)
    df[label_col] = df[label_col].astype(str).fillna("").map(normalize_text)
    df = df[(df[text_col] != "") & (df[label_col] != "")].reset_index(drop=True)

    if len(df) < 5:
        raise ValueError("❌ 학습할 데이터가 너무 적어요. (최소 5개 이상 권장)")

    X = df[text_col].tolist()
    y = df[label_col].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if len(set(y)) > 1 else None
    )

    # 모델 (다중클래스는 scikit-learn이 자동 처리)
    clf = LogisticRegression(max_iter=3000)  # multi_class 파라미터 제거(버전 이슈 방지)
    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", clf),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    labels = sorted(list(set(y_train) | set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report = classification_report(y_test, y_pred, labels=labels, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": acc,
        "labels": labels,
        "cm": cm,
        "report": report,
    }
    return pipe, metrics


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="리뷰 감정 분석", layout="wide")
st.title("리뷰 감정 분석 (만족 / 중립 / 부정)")

ROOT = Path(__file__).resolve().parent
DEFAULT_REVIEW_PATH = ROOT / "Review.xlsx"
DEFAULT_KEYWORD_PATH = ROOT / "Keyword.xlsx"

st.sidebar.header("1) 데이터 로드")

use_uploaded = st.sidebar.checkbox("업로드한 파일로 대체하기 (선택)", value=False)

review_file = None
keyword_file = None

if use_uploaded:
    review_file = st.sidebar.file_uploader("Review.xlsx 업로드", type=["xlsx"])
    keyword_file = st.sidebar.file_uploader("Keyword.xlsx 업로드", type=["xlsx"])
else:
    # 레포에 있는 파일을 자동으로 사용
    if DEFAULT_REVIEW_PATH.exists() and DEFAULT_KEYWORD_PATH.exists():
        review_file = str(DEFAULT_REVIEW_PATH)
        keyword_file = str(DEFAULT_KEYWORD_PATH)
        st.sidebar.success("✅ 레포에 업로드된 Review.xlsx / Keyword.xlsx 자동 로드!")
    else:
        st.sidebar.warning(
            "레포에 Review.xlsx / Keyword.xlsx가 없어서 업로드가 필요해요.\n"
            "→ 깃허브 루트에 파일이 있는지 확인!"
        )
        review_file = st.sidebar.file_uploader("Review.xlsx 업로드", type=["xlsx"])
        keyword_file = st.sidebar.file_uploader("Keyword.xlsx 업로드", type=["xlsx"])

# 실제 로드
if not review_file or not keyword_file:
    st.info("왼쪽 사이드바에서 파일을 로드해줘.")
    st.stop()

try:
    df_review = pd.read_excel(review_file)
    df_kw = pd.read_excel(keyword_file)
except Exception as e:
    st.error(f"엑셀을 읽는 중 오류: {e}")
    st.stop()

st.sidebar.header("2) 컬럼 설정")

# 기본 추천 컬럼명 후보
review_cols = list(df_review.columns)
default_text = "review" if "review" in [c.lower() for c in review_cols] else review_cols[0]
default_label = "label" if "label" in [c.lower() for c in review_cols] else (review_cols[1] if len(review_cols) > 1 else review_cols[0])

text_col = st.sidebar.selectbox("리뷰 텍스트 컬럼", options=review_cols, index=review_cols.index(default_text) if default_text in review_cols else 0)
label_col = st.sidebar.selectbox("라벨 컬럼", options=review_cols, index=review_cols.index(default_label) if default_label in review_cols else min(1, len(review_cols)-1))

st.sidebar.header("3) 학습 설정")
test_size = st.sidebar.slider("테스트 비율", 0.1, 0.5, 0.2, 0.05)
train_btn = st.sidebar.button("모델 학습 & 평가", type="primary")

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

# Predict on all rows
st.subheader("전체 리뷰 예측 결과")
texts_all = df_review[text_col].astype(str).fillna("").tolist()
pred = model.predict(texts_all)

# predict_proba 지원 여부 체크
proba_df = None
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(texts_all)
    proba_df = pd.DataFrame(proba, columns=[f"p_{c}" for c in model.named_steps["clf"].classes_])

out_df = df_review.copy()
out_df["Pred_Sentiment"] = pred
if proba_df is not None:
    out_df = pd.concat([out_df, proba_df], axis=1)

st.dataframe(out_df.head(30), use_container_width=True)

# Keyword hit analysis
st.subheader("키워드 히트 Top (Keyword.xlsx 기준)")
try:
    sent_lex, _ = build_lexicon(df_kw)
    rows = []
    for sent, kws in sent_lex.items():
        for kw in kws:
            if not kw:
                continue
            cnt = sum(1 for t in texts_all if kw in str(t))
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
            top_by = kw_hits.groupby("Sentiment").head(10)
            st.dataframe(top_by, use_container_width=True)
    else:
        st.info("데이터에서 Keyword.xlsx 키워드 매칭이 거의 없어요. (표현/띄어쓰기/키워드 확장 추천)")
except Exception as e:
    st.warning(f"키워드 분석을 건너뛰었어요: {e}")

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
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
