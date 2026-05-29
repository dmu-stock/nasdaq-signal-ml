from app.collector.news_crawler import fetch_all_news
from app.collector.headline_preprocessor import preprocess_headlines_batch
from app.services.finbert_service import  FinBertSentimentAnalyzer
from datetime import datetime

class NewsSentimentPipeline:

    def __init__(self):
         self.analyzer = FinBertSentimentAnalyzer()
    
    def run(self):
        df_news = fetch_all_news()

        raw_head = df_news["headline"].tolist()
        clean_head = df_news["clean_headline"].tolist()
        

        final_scores, raw_results = self.analyzer.analyze_hybrid_batch(raw_head, clean_head)
        df_news["sentiment_score"] = final_scores
        
        # (선택 사항) 분석 라벨이나 확신도는 '날것' 기준으로 남겨두는 게 일반적입니다.
        df_news["label"] = [r["label"] for r in raw_results]
        df_news["confidence"] = [r["confidence"] for r in raw_results]

        return df_news
         

if __name__ == "__main__":

    pipeline = NewsSentimentPipeline()

    result_df = pipeline.run()

    print(result_df.head())

    # =========================
    # CSV 저장 추가
    # =========================
    today = datetime.now().strftime("%Y%m%d")
    result_df.to_csv(
        f"news_sentiment_{today}.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n[저장 완료] news_sentiment_results.csv")