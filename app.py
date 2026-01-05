# app.py
# Streamlit: 리뷰 감정 분석 (만족/중립/부정) — 로컬에 있는 Review.xlsx / Keyword.xlsx 자동 로드
# - Streamlit Cloud / 로컬 모두 대응: 파일이 없으면 업로더로 대체
# - 멀티클래스 에러 해결: OneVsRestClassifier + liblinear
# - 그래프는 제거(요청 반영), 대신 표/텍스트로 확인

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------
# 기본 설정
# ---------------------------
st.set_page_config(page_title="리뷰 감정 분석 (만족/중립/부정)", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DEFAULT_REVIEW_PATH = APP_DIR / "Review.xlsx"
DEFAULT_KEYWORD_PATH = APP_DIR / "Keyword.xlsx"

DEFAULT_LABEL_ORDER = ["Positive", "Neutral", "Negative"]  # 내부 표준 라벨
KOREAN_MAP = {"긍정": "Positive", "중립": "Neutral", "부정": "Negative"}
EN_MAP = {"positive": "Positive", "neutral": "Neutral", "negative": "Negative"}

# ---------------------------
# 유틸
# ---------------------------
def normalize_label(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s:
        return ""
    # 한국어
    if s in KOREAN_MAP:
        return KOREAN_MAP[s]
    # 영어
    sl = s.lower()
    if sl in EN_MAP:
        return EN_MAP[sl]
    # 혹시 '만족/중립/부정' 같은 표현이 들어오면 대충 매핑
    if "만족" in s or "긍정" in s:
        return "Positive"
    if "중립" in s:
        return "Neutral"
    if "불만" in s or "부정" in s:
        return "Negative"
    return s  # 그대로 (사용자 라벨도 허용)

def smart_find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def clean_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u200b", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def explode_keywords_cell(cell) -> List[str]:
    """Keywords 셀 하나에서 키워드 리스트로 분해."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s:
        return []
    # 구분자 다양하게 처리: , / ; | \n
    parts = re.split(r"[,\n;/|]+", s)
    out = [p.strip() for p in parts if p.strip()]
    return out

# ---------------------------
# 데이터 로드
# ---------------------------
@st.cache_data(show_spinner=False)
def load_excel_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def load_excel_path(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def load_review_and_keyword() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], str]:
    """
    1) 리포지토리에 Review.xlsx/Keyword.xlsx가 있으면 자동 로드
    2) 없으면 업로더로 받도록 안내
    """
    msg = ""
    review_df = None
    kw_df = None

    if DEFAULT_REVIEW_PATH.exists():
        try:
            review_df = load_excel_path(str(DEFAULT_REVIEW_PATH))
            msg += f"✅ 로컬 파일 로드: {DEFAULT_REVIEW_PATH.name}\n"
        except Exception as e:
            msg += f"❌ Review.xlsx 로드 실패: {e}\n"

    if DEFAULT_KEYWORD_PATH.exists():
        try:
            kw_df = load_excel_path(str(DEFAULT_KEYWORD_PATH))
            msg += f"✅ 로컬 파일 로드: {DEFAULT_KEYWORD_PATH.name}\n"
        except Exception as e:
            msg += f"❌ Keyword.xlsx 로드 실패: {e}\n"

    return review_df, kw_df, msg.strip()

# ---------------------------
# 모델
# ---------------------------
@dataclass
class SentimentModel:
    vectorizer: TfidfVectorizer
    clf: OneVsRestClassifier
    label_order: List[str]

    def predict(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        X = self.vectorizer.transform(texts)
        # OneVsRestClassifier는 predict_proba 지원(대부분 버전에서)
        proba = self.clf.predict_proba(X)
        pred = self.clf.predict(X)
        # classes_ 접근 안전하게
        classes = getattr(self.clf, "classes_", None)
        if classes is None:
            # 드물게 estimator에만 있는 경우
            classes = getattr(self.clf, "estimators_", [None])[0].classes_  # type: ignore
        classes = list(classes)
        return pred, proba, classes

def train_model(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    test_size: float = 0.25,
    seed: int = 42,
) -> Tuple[SentimentModel, Dict]:
    # 정리
    work = df[[text_col, label_col]].copy()
    work[text_col] = work[text_col].apply(clean_text)
    work[label_col] = work[label_col].apply(normalize_label)

    work = work[(work[text_col].str.len() > 0) & (work[label_col].str.len() > 0)].copy()
    if work.empty:
        raise ValueError("학습할 데이터가 비어있어요. (텍스트/라벨 컬럼 확인)")

    # 라벨 3개 이상인지 확인
    labels = sorted(work[label_col].unique().tolist())
    if len(labels) < 2:
        raise ValueError(f"라벨 종류가 너무 적어요: {labels}")

    # split (가능하면 stratify)
    y = work[label_col].astype(str).values
    X_text = work[text_col].astype(str).values

    strat = y if len(np.unique(y)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_text, y, test_size=test_size, random_state=seed, stratify=strat
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_tr_vec = vectorizer.fit_transform(X_tr)
    X_te_vec = vectorizer.transform(X_te)

    # ✅ 멀티클래스 안전 조합 (Streamlit Cloud/구버전 대응)
    base = LogisticRegression(max_iter=3000, solver="liblinear")
    clf = OneVsRestClassifier(base)
    clf.fit(X_tr_vec, y_tr)

    y_pred = clf.predict(X_te_vec)

    metrics = {
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "labels": sorted(list(set(y))),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "report": classification_report(y_te, y_pred, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y_te, y_pred, labels=sorted(list(set(y)))),
        "confusion_labels": sorted(list(set(y))),
    }

    model = SentimentModel(vectorizer=vectorizer, clf=clf, label_order=sorted(list(set(y))))
    return model, metrics

# ---------------------------
# 키워드 테이블(그래프 없이)
# ---------------------------
def build_keyword_table(kw_df: pd.DataFrame, sentiment_col: str, keywords_col: str) -> pd.DataFrame:
    tmp = kw_df[[sentiment_col, keywords_col]].copy()
    tmp[sentiment_col] = tmp[sentiment_col].apply(normalize_label)

    rows = []
    for _, r in tmp.iterrows():
        senti = r[sentiment_col]
        kws = explode_keywords_cell(r[keywords_col])
        for k in kws:
            rows.append((senti, k))
    out = pd.DataFrame(rows, columns=["Sentiment", "Keyword"])
    out["Keyword"] = out["Keyword"].astype(str).str.strip()
    out = out[(out["Sentiment"].str.len() > 0) & (out["Keyword"].str.len() > 0)].copy()
    return out

def keyword_hit_counts(review_df: pd.DataFrame, text_col: str, kw_table: pd.DataFrame) -> pd.DataFrame:
    """
    리뷰 텍스트에서 Keyword가 등장하는 횟수(아주 단순 포함 기준)를 집계해서 표로 반환.
    """
    texts = review_df[text_col].astype(str).apply(clean_text).tolist()

    # 속도 개선: 키워드별 정규식 준비
    counts = []
    for senti in sorted(kw_table["Sentiment"].unique().tolist()):
        sub = kw_table[kw_table["Sentiment"] == senti]["Keyword"].tolist()
        for k in sub:
            # 단순 포함 (공백/기호 많은 경우도 대응)
            pattern = re.escape(k)
            c = 0
            for t in texts:
                if re.search(pattern, t, flags=re.IGNORECASE):
                    c += 1
            counts.append((senti, k, c))

    out = pd.DataFrame(counts, columns=["Sentiment", "Keyword", "HitCount"])
    out = out.sort_values(["Sentiment", "HitCount", "Keyword"], ascending=[True, False, True]).reset_index(drop=True)
    return out

# ---------------------------
# UI
# ---------------------------
st.title("리뷰 감정 분석 (만족 / 중립 / 부정)")
st.caption("✅ Review.xlsx / Keyword.xlsx를 같은 폴더(app.py 옆)에 두면 자동으로 읽어요. (없으면 업로드 방식으로도 가능)")

review_df, kw_df, load_msg = load_review_and_keyword()
if load_msg:
    st.info(load_msg)

# 업로더 (자동 로드 실패 시에만 사용)
with st.sidebar:
    st.header("1) 데이터 소스")
    st.write("자동 로드가 안 될 때만 업로드로 대체하세요.")

    up_review = None
    up_kw = None

    if review_df is None:
        up_review = st.file_uploader("Review.xlsx 업로드", type=["xlsx"], key="review_uploader")
        if up_review is not None:
            review_df = load_excel_bytes(up_review.getvalue())

    if kw_df is None:
        up_kw = st.file_uploader("Keyword.xlsx 업로드", type=["xlsx"], key="kw_uploader")
        if up_kw is not None:
            kw_df = load_excel_bytes(up_kw.getvalue())

# 데이터 유효성 체크
if review_df is None:
    st.warning("Review.xlsx를 찾지 못했어. app.py 옆에 Review.xlsx를 두거나, 왼쪽에서 업로드해줘.")
    st.stop()

# 컬럼 자동 탐지
default_text_col = smart_find_col(review_df, ["review", "text", "리뷰", "내용", "comment"])
default_label_col = smart_find_col(review_df, ["sentiment", "label", "감정", "라벨", "분류"])

# 사이드바: 컬럼 설정
with st.sidebar:
    st.header("2) 컬럼 설정")
    text_col = st.selectbox(
        "리뷰 텍스트 컬럼",
        options=list(review_df.columns),
        index=list(review_df.columns).index(default_text_col) if default_text_col in review_df.columns else 0,
    )
    label_col = st.selectbox(
        "감정 라벨 컬럼",
        options=list(review_df.columns),
        index=list(review_df.columns).index(default_label_col) if default_label_col in review_df.columns else 0,
    )
    test_size = st.slider("테스트 비율", 0.1, 0.5, 0.25, 0.05)
    seed = st.number_input("랜덤 시드", value=42, step=1)

# 메인: 탭
tab1, tab2, tab3 = st.tabs(["데이터 확인", "학습/평가", "리뷰 한 줄 예측"])

with tab1:
    st.subheader("Review.xlsx 미리보기")
    st.dataframe(review_df.head(20), use_container_width=True)

    if kw_df is not None:
        st.subheader("Keyword.xlsx 미리보기")
        st.dataframe(kw_df.head(20), use_container_width=True)
    else:
        st.info("Keyword.xlsx는 없어도 학습/예측은 가능해. (키워드 히트 표만 스킵됨)")

with tab2:
    st.subheader("모델 학습/평가")

    if st.button("학습 시작", type="primary"):
        try:
            with st.spinner("학습 중... (TF-IDF + LogisticRegression One-vs-Rest)"):
                model, metrics = train_model(
                    df=review_df,
                    text_col=text_col,
                    label_col=label_col,
                    test_size=float(test_size),
                    seed=int(seed),
                )
            st.session_state["model"] = model
            st.session_state["metrics"] = metrics
            st.success("학습 완료!")
        except Exception as e:
            st.error(f"학습 실패: {e}")

    if "metrics" in st.session_state:
        m = st.session_state["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Train 샘플 수", m["n_train"])
        c2.metric("Test 샘플 수", m["n_test"])
        c3.metric("Accuracy", f"{m['accuracy']:.3f}")

        st.write("라벨 종류:", ", ".join(map(str, m["labels"])))

        # classification report (표)
        st.markdown("#### 분류 리포트 (표)")
        rep = pd.DataFrame(m["report"]).T
        st.dataframe(rep, use_container_width=True)

        # confusion matrix (표)
        st.markdown("#### Confusion Matrix (표)")
        cm = pd.DataFrame(
            m["confusion"],
            index=[f"true_{x}" for x in m["confusion_labels"]],
            columns=[f"pred_{x}" for x in m["confusion_labels"]],
        )
        st.dataframe(cm, use_container_width=True)

        # Keyword 히트 표(그래프 없이)
        st.markdown("#### 키워드 히트 Top (Keyword.xlsx 기준)")
        if kw_df is None:
            st.info("Keyword.xlsx가 없어서 키워드 히트 분석은 건너뛰었어.")
        else:
            # Keywords 컬럼 지원(요청 반영)
            default_kw_sent_col = smart_find_col(kw_df, ["sentiment", "label", "감정", "라벨"])
            default_kw_keywords_col = smart_find_col(kw_df, ["keywords", "keyword", "키워드"])

            # 만약 자동탐지가 실패하면, 사용자가 선택 가능
            kw_sent_col = st.selectbox(
                "Keyword.xlsx 감정 컬럼",
                options=list(kw_df.columns),
                index=list(kw_df.columns).index(default_kw_sent_col) if default_kw_sent_col in kw_df.columns else 0,
                key="kw_sent_col",
            )
            kw_keywords_col = st.selectbox(
                "Keyword.xlsx 키워드 컬럼",
                options=list(kw_df.columns),
                index=list(kw_df.columns).index(default_kw_keywords_col) if default_kw_keywords_col in kw_df.columns else 0,
                key="kw_keywords_col",
            )

            try:
                kw_table = build_keyword_table(kw_df, kw_sent_col, kw_keywords_col)
                if kw_table.empty:
                    st.warning("Keyword.xlsx에서 키워드를 읽었는데 비어있어. (구분자/셀 값 확인)")
                else:
                    hits = keyword_hit_counts(review_df, text_col, kw_table)
                    top_n = st.slider("표시할 Top N", 5, 50, 15, 5)
                    # 감정별 top n
                    out_frames = []
                    for senti in sorted(hits["Sentiment"].unique().tolist()):
                        sub = hits[hits["Sentiment"] == senti].head(int(top_n))
                        out_frames.append(sub)
                    out = pd.concat(out_frames, axis=0).reset_index(drop=True)
                    st.dataframe(out, use_container_width=True)
            except Exception as e:
                st.error(f"키워드 히트 분석 실패: {e}")

with tab3:
    st.subheader("리뷰 한 줄 입력 → 감정 예측")

    if "model" not in st.session_state:
        st.info("먼저 [학습/평가] 탭에서 학습을 완료해줘.")
    else:
        model: SentimentModel = st.session_state["model"]

        user_text = st.text_area(
            "리뷰를 입력하세요",
            height=120,
            placeholder="예) 기사님이 너무 친절하고 시간도 정확했어요!",
        )

        if st.button("예측하기"):
            t = clean_text(user_text)
            if not t:
                st.warning("텍스트를 입력해줘!")
            else:
                pred, proba, classes = model.predict([t])
                pred_label = str(pred[0])

                # proba는 classes 순서로 나옴
                proba_row = proba[0]
                proba_df = pd.DataFrame({"label": classes, "prob": proba_row}).sort_values("prob", ascending=False)

                st.success(f"예측 결과: **{pred_label}**")
                st.dataframe(proba_df, use_container_width=True)

                st.caption("Tip: 정확도가 낮으면 Review.xlsx의 라벨 품질/데이터 수가 제일 크게 영향을 줘요.")

# 끝
