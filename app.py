import streamlit as st
import pandas as pd
import numpy as np
from sentiment_model import train_model, analyze_keywords, predict_sentiment
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="리뷰 감정 분석", layout="wide")

st.title("📊 리뷰 감정 분석 (Streamlit)")

# =========================
# 1. 엑셀 파일 로드
# =========================
st.sidebar.header("📁 데이터 로드")

use_uploaded = st.sidebar.checkbox("엑셀 파일 직접 업로드", value=False)

if use_uploaded:
    review_file = st.sidebar.file_uploader("Review.xlsx 업로드", type=["xlsx"])
    keyword_file = st.sidebar.file_uploader("Keyword.xlsx 업로드", type=["xlsx"])

    if review_file is None or keyword_file is None:
        st.warning("리뷰 엑셀과 키워드 엑셀을 모두 업로드해주세요.")
        st.stop()

    review_df = pd.read_excel(review_file)
    keyword_df = pd.read_excel(keyword_file)

else:
    # 👉 GitHub에 같이 올린 엑셀을 읽는 부분 (중요)
    review_df = pd.read_excel("Review.xlsx")
    keyword_df = pd.read_excel("Keyword.xlsx")

st.success("✅ 엑셀 데이터 로드 완료")

# =========================
# 2. 컬럼 선택
# =========================
st.sidebar.header("🧩 컬럼 설정")

text_col = st.sidebar.selectbox(
    "리뷰 텍스트 컬럼",
    review_df.columns
)

label_col = st.sidebar.selectbox(
    "감정 라벨 컬럼",
    review_df.columns
)

# =========================
# 3. 데이터 미리보기
# =========================
st.subheader("📄 리뷰 데이터 미리보기")
st.dataframe(review_df.head())

# =========================
# 4. 모델 학습
# =========================
st.subheader("🤖 모델 학습")

test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2)

if st.button("모델 학습 시작"):
    with st.spinner("모델 학습 중..."):
        model, metrics = train_model(
            review_df,
            keyword_df,
            text_col,
            label_col,
            test_size=test_size,
            seed=42
        )

    st.success("✅ 모델 학습 완료")

    st.write("### 📈 모델 성능")
    st.json(metrics)

# =========================
# 5. 감정 분포 시각화
# =========================
st.subheader("📊 감정 분포")

fig, ax = plt.subplots()
sns.countplot(x=review_df[label_col], ax=ax)
ax.set_title("감정 라벨 분포")
st.pyplot(fig)

# =========================
# 6. 키워드 분석
# =========================
st.subheader("🔑 키워드 분석")

keyword_result = analyze_keywords(review_df, keyword_df, text_col)

fig2, ax2 = plt.subplots(figsize=(8, 4))
sns.barplot(
    x=keyword_result.values,
    y=keyword_result.index,
    ax=ax2
)
ax2.set_title("감정 키워드 등장 빈도")
st.pyplot(fig2)

# =========================
# 7. 단일 리뷰 예측
# =========================
st.subheader("✏️ 리뷰 감정 예측")

user_review = st.text_area("리뷰를 입력하세요")

if user_review and st.button("감정 예측"):
    pred = predict_sentiment(user_review)
    st.info(f"예측된 감정: **{pred}**")
