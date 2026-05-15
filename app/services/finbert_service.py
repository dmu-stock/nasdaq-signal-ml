from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
import numpy as np


class FinBertSentimentAnalyzer:
    def __init__(self):
        self.model_name = "yiyanghkust/finbert-tone"

        # 모델 로드
        self.finbert = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=3
        )

        # 토크나이저 로드
        self.tokenizer = BertTokenizer.from_pretrained(
            self.model_name
        )

        # 파이프라인 생성
        self.nlp = pipeline(
            "sentiment-analysis",
            model=self.finbert,
            tokenizer=self.tokenizer
        )


    def analyze(self, headlines: list[str]):
        results = self.nlp(headlines)

        analyzed_results = []

        for r in results:
            label = r["label"]
            confidence = r["score"]

            # 방향 + 강도 점수
            if label == "Positive":
                sentiment_score = confidence

            elif label == "Negative":
                sentiment_score = -confidence

            else:  # Neutral
                sentiment_score = confidence * 0.15

            analyzed_results.append({
                "label": label,
                "confidence": confidence,
                "sentiment_score": round(sentiment_score, 4)
            })

        return analyzed_results
    
    def analyze_hybrid_batch(self, raw_headlines: list[str], clean_headlines: list[str]):
        """날것(8)과 정제본(2)의 가중 평균 점수 계산"""
        raw_results = self.analyze(raw_headlines)
        clean_results = self.analyze(clean_headlines)
        
        final_scores=[]
        for r, c in zip(raw_results, clean_results):

            if c["label"] == "Neutral":
                score = r["sentiment_score"]
            else:
                score = (r["sentiment_score"] * 0.8) + (c["sentiment_score"] * 0.2)
            final_scores.append(round(score, 4))
        
        return final_scores, raw_results
