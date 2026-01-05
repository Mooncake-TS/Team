# app.py
# Streamlit - 리뷰 감정 분석 (만족/중립/부정) + 키워드 시각화 + 단일 리뷰 예측
# - Repo에 Review.xlsx / Keyword.xlsx가 있으면 자동 로드
# - 없으면 업로드 UI 표시
# - Keyword.xlsx 컬럼: Sentiment / Keywords (또는 Keyword) 자동 인식

import io
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# -------------------------
# 기본 설정
# -------------------------
st.set_page_config(page_title="리뷰 감정 분석", layout="wide")

DEFAULT_REVIEW_PATH = Path("Review.xlsx")
DEFAULT_KEYWORD_PATH = Path("Keyword.xlsx")

# -------------------------
# 유틸
# -------------------------
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None


def load_excel_from_repo_or_upload(label: str, default_path: Path) -> pd.DataFrame | None:
    """
    Repo에 파일 있으면 그걸 읽고, 없으면 업로드 위젯을 보여준다.
    """
    if default_path.exists():
        try:
            return pd.read_excel(default_path)
        except Exception as e:
            st.error(f"❌ {default_path} 읽기 실패: {e}")
            return None

    st.warning(f"📌 리포지토리에 `{default_path.name}` 파일이 없어서 업로드가 필요해요.")
    up = st.file_uploader(label, type=["xlsx"])
    if up is None:
        return None
    try:
        return pd.read_excel(up)
    except Exception as e:
        st.error(f"❌ 업로드 파일 읽기 실패: {e}")
        return None


def build_lexicon(df_kw: pd.DataFrame) -> Tuple[Dict[str, List[str]], List[str], str, str]:
    """
    Keyword.xlsx에서 감정별 키워드 사전을 만든다.
    컬럼 자동 인식:
      - 감정: Sentiment / 감정 / label / 라벨
      - 키워드: Keywords / Keyword / 키워드
    """
    sentiment_col = find_col(df_kw, ["Sentiment", "sentiment", "감정", "label", "라벨"])
    keyword_col = find_col(df_kw, ["Keywords", "keywords", "Keyword", "keyword", "키워드"])

    if sentiment_col is None or keyword_col is None:
        raise ValueError(
            f"Keyword.xlsx 컬럼을 찾지 못했어요. "
            f"현재 컬럼: {list(df_kw.columns)} / "
            f"필요 예: Sentiment(감정), Keywords(키워드)"
        )

    df = df_kw[[sentiment_col, keyword_col]].copy()
    df[sentiment_col] = df[sentiment_col].astype(str).map(normalize_text)
    df[keyword_col] = df[keyword_col].astype(str).map(normalize_text)

    # Keywords 컬럼이 "키워드1,키워드2,..." 형태일 수 있어서 분해
    lex: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        sent = row[sentiment_col]
        kws_raw = row[keyword_col]
        if not sent or not kws_raw:
            continue

        # 구분자: 쉼표/슬래시/세미콜론/파이프 등 대응
        parts = re.split(r"[,\|/;]+", kws_raw)
        parts = [p.strip() for p in parts if p.strip()]
        if not parts:
            continue

        lex.setdefault(sent, [])
        lex[sent].extend(parts)

    # 중복 제거(순서 유지)
    for k in list(lex.keys()):
        seen = set()
        uniq = []
        for w in lex[k]:
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        lex[k] = uniq

    sentiments = sorted(list(lex.keys()))
    return lex, sentiments, sentiment_col, keyword_col


def count_keyword_hits(texts: List[str], lex: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for sent, kws in lex.items():
        for kw in kws:
            kw = normalize_text(kw)
            if not kw:
                continue
            cnt = sum(1 for t in texts if kw in t)
            if cnt:
                rows.append((sent, kw, cnt))
    df = pd.DataFrame(rows, columns=["Sentiment", "Keyword", "Count"])
    if df.empty:
        return df
    return df.sort_values(["Sentiment", "Count"], ascending=[True, False]).reset_index(drop=True)


def plot_top_keywords(kw_hits: pd.DataFrame, top_n: int = 15):
    if kw_hits.empty:
        st.info("키워드 매칭 결과가 거의 없어요. Keyword.xlsx의 키워드를 더 늘리면 훨씬 잘 나와요.")
        return

    sentiments = kw_hits["Sentiment"].unique().tolist()
    preferred_order = ["만족", "긍정", "중립", "부정"]
    sentiments = sorted(sentiments, key=lambda x: preferred_order.index(x) if x in preferred_order else 999)

    fig, axes = plt.subplots(1, len(sentiments), figsize=(6 * len(sentiments), 4))
    if len(sentiments) == 1:
        axes = [axes]

    for ax, sent in zip(axes, sentiments):
        sub = kw_hits[kw_hits["Sentiment"] == sent].sort_values("Count", ascending=False).head(top_n)
        ax.barh(sub["Keyword"][::-1], sub["Count"][::-1])
        ax.set_title(f"{sent} 키워드 Top {top_n}")
        ax.set_xlabel("Count")
        ax.set_ylabel("Keyword")

    plt.tight_layout()
    st.pyplot(fig)


def train_model(df_review: pd.DataFrame, text_col: str, label_col: str, test_size: float = 0.2, seed: int = 42):
    X = df_review[text_col].astype(str).map(normalize_text)
    y = df_review[label_col].astype(str).map(normalize_text)

    # 빈값 제거
    mask = (X != "") & (y != "")
    X = X[mask]
    y = y[mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if y.nunique() > 1 else None
    )

    # ⚠️ multi_class 파라미터는 환경에 따라 에러 나서 제거(너가 겪은 그 오류 방지)
    clf = LogisticRegression(max_iter=3000)

    model = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", clf),
        ]
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "report": classification_report(y_test, pred, output_dict=False),
        "confusion": confusion_matrix(y_test, pred).tolist(),
        "labels": sorted(y.unique().tolist()),
        "test_size": len(X_test),
        "train_size": len(X_train),
    }
    return model, metrics, (X_test, y_test, pred)


# -------------------------
# UI
# -------------------------
st.title("리뷰 감정 분석 (만족 / 중립 / 부정)")

left, right = st.columns([1, 3])

with left:
    st.header("데이터 로드")
    st.caption("Repo에 Review.xlsx, Keyword.xlsx가 있으면 자동으로 읽고, 없으면 업로드 UI가 떠요.")

    df_review = load_excel_from_repo_or_upload("Review.xlsx 업로드", DEFAULT_REVIEW_PATH)
    df_kw = load_excel_from_repo_or_upload("Keyword.xlsx 업로드", DEFAULT_KEYWORD_PATH)

    st.divider()
    st.header("설정")

    test_size = st.slider("테스트 비율", 0.1, 0.5, 0.2, 0.05)
    seed = st.number_input("랜덤 시드", value=42, step=1)

with right:
    if df_review is None or df_kw is None:
        st.info("왼쪽에서 Review.xlsx / Keyword.xlsx를 준비하면 여기서 분석이 진행돼요.")
        st.stop()

    st.subheader("데이터 미리보기")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Review.xlsx (상위 10개)")
        st.dataframe(df_review.head(10), use_container_width=True)
    with c2:
        st.caption("Keyword.xlsx (상위 10개)")
        st.dataframe(df_kw.head(10), use_container_width=True)

    # Review 컬럼 자동 추정
    text_col_guess = find_col(df_review, ["Review", "review", "리뷰", "text", "텍스트", "내용"])
    label_col_guess = find_col(df_review, ["Sentiment", "sentiment", "감정", "label", "라벨"])

    st.divider()
    st.subheader("컬럼 선택")
    colA, colB = st.columns(2)
    with colA:
        text_col = st.selectbox("리뷰 텍스트 컬럼", options=list(df_review.columns), index=(list(df_review.columns).index(text_col_guess) if text_col_guess in df_review.columns else 0))
    with colB:
        label_col = st.selectbox("감정 라벨 컬럼", options=list(df_review.columns), index=(list(df_review.columns).index(label_col_guess) if label_col_guess in df_review.columns else 0))

    # -------------------------
    # 키워드 분석 (Keywords 지원!)
    # -------------------------
    st.divider()
    st.subheader("키워드 분석 (Keyword.xlsx 기준)")

    try:
        sent_lex, sentiments_list, s_col, k_col = build_lexicon(df_kw)

        texts_all = [normalize_text(t) for t in df_review[text_col].astype(str).fillna("").tolist()]
        kw_hits = count_keyword_hits(texts_all, sent_lex)

        cc1, cc2 = st.columns(2)
        with cc1:
            st.caption("전체 Top 30")
            if not kw_hits.empty:
                st.dataframe(kw_hits.sort_values("Count", ascending=False).head(30), use_container_width=True)
            else:
                st.write("매칭 결과 없음")
        with cc2:
            st.caption("감정별 Top 10")
            if not kw_hits.empty:
                st.dataframe(kw_hits.groupby("Sentiment").head(10), use_container_width=True)
            else:
                st.write("매칭 결과 없음")

        st.subheader("감정별 키워드 시각화")
        top_n = st.slider("그래프에 표시할 키워드 개수(감정별)", 5, 30, 15, 1)
        plot_top_keywords(kw_hits, top_n=top_n)

    except Exception as e:
        st.warning(f"키워드 분석을 건너뛰었어요. ❌ {e}")

    # -------------------------
    # 모델 학습
    # -------------------------
    st.divider()
    st.subheader("모델 학습/평가")

    if st.button("학습 실행"):
        with st.spinner("학습 중..."):
            model, metrics, test_pack = train_model(df_review, text_col, label_col, test_size=test_size, seed=int(seed))

        st.success("✅ 학습 완료!")
        st.write(f"- Train: {metrics['train_size']}개 / Test: {metrics['test_size']}개")
        st.write(f"- Accuracy: **{metrics['accuracy']:.4f}**")

        st.caption("분류 리포트")
        st.code(metrics["report"])

        # -------------------------
        # 단일 리뷰 입력 → 예측 (요청사항 2번)
        # -------------------------
        st.divider()
        st.subheader("리뷰 한 줄 입력 → 감정 예측")

        user_text = st.text_area(
            "리뷰를 입력하세요",
            placeholder="예) 기사님이 너무 친절하고 시간도 정확했어요!",
            height=120,
        )

        if st.button("예측하기"):
            txt = normalize_text(user_text)
            if not txt:
                st.warning("리뷰 내용을 입력해줘!")
            else:
                pred_label = model.predict([txt])[0]
                st.write(f"### 예측 결과: **{pred_label}**")

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba([txt])[0]
                    classes = model.named_steps["clf"].classes_
                    prob_df = (
                        pd.DataFrame({"label": classes, "prob": probs})
                        .sort_values("prob", ascending=False)
                        .reset_index(drop=True)
                    )
                    st.caption("라벨별 확률")
                    st.dataframe(prob_df, use_container_width=True)
                    st.bar_chart(prob_df.set_index("label")["prob"])

        st.caption("Tip: 정확도가 낮으면 Review.xlsx 라벨 품질/데이터 수가 제일 크게 영향을 줘요.")
    else:
        st.info("위에서 컬럼을 고른 뒤, '학습 실행' 버튼을 눌러줘.")
