# 리뷰 감정 분석 (Streamlit)

## 실행
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 입력 파일 형식
### Review.xlsx
- Review : 리뷰 텍스트
- Sentiment : 라벨 (Positive / Neutral / Negative)

### Keyword.xlsx
- Sentiment : Positive / Neutral / Negative
- Category  : (선택) 카테고리명
- Keywords  : 쉼표로 구분된 키워드 목록
- Rationale : (선택) 설명