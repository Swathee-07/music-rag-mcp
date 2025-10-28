

import streamlit as st
import json
from typing import Dict
from sklearn.metrics.pairwise import cosine_similarity
from rouge import Rouge
import requests
import os
from dotenv import load_dotenv

load_dotenv()
QUERIES_PATH = 'evaluation/queries.json'
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
EVAL_MODEL = 'llama-3.1-8b-instant'

@st.cache_resource
def get_eval_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_ground_truth():
    try:
        file_path = "evaluation/queries.json"  # Update this path
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return []

def find_ground_truth(user_query: str) -> str:
    """Finds ground truth, ignoring case and whitespace."""
    try:
        with open(QUERIES_PATH, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        normalized_query = user_query.strip().lower()
        for item in queries:
            if item.get('query', '').strip().lower() == normalized_query:
                return item.get('answer', '')
        return ""
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Evaluation file error: {e}")
        return ""

def call_llm_as_judge(truth: str, pred: str) -> float:
    """Uses Groq API to get a semantic similarity score from an LLM."""
    if not truth or not pred or not GROQ_API_KEY:
        return 0.0
        
    prompt = f"""
As an impartial judge, rate how well the 'Predicted Answer' matches the 'Ground-Truth Answer' on a scale from 0.0 to 1.0.
- 1.0 means a perfect match or semantically identical.
- 0.0 means no relevant information.
Return only a single float number.

Ground-Truth Answer: "{truth}"
Predicted Answer: "{pred}"

Score:
""".strip()
    
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={"model": EVAL_MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 5, "temperature": 0.0}
        )
        resp.raise_for_status()
        score_str = resp.json()['choices'][0]['message']['content']
        return float(score_str)
    except Exception as e:
        print(f"LLM Judge call failed: {e}")
        return 0.0

def all_metrics(user_query: str, model_answer: str) -> Dict[str, float]:
    """Calculates all specified evaluation metrics."""
    truth = find_ground_truth(user_query)
    if not truth or not model_answer or model_answer.startswith("Error"):
        return {metric: 0.0 for metric in ["f1", "precision", "recall", "bertscore", "cosine", "rougeL", "f1_llm_combined"]}

    # F1, Precision, Recall
    pred_tokens = set(model_answer.lower().split())
    truth_tokens = set(truth.lower().split())
    common = pred_tokens.intersection(truth_tokens)
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(truth_tokens) if truth_tokens else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    # Cosine Similarity
    embedder = get_eval_model()
    embeddings = embedder.encode([truth, model_answer])
    cosine = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    # ROUGE-L
    rouge = Rouge()
    try:
        rouge_scores = rouge.get_scores(model_answer, truth)
        rougeL = rouge_scores[0]['rouge-l']['f']
    except (ValueError, KeyError):
        rougeL = 0.0
        
    # LLM-as-a-judge and Combined Score
    llm_judge_score = call_llm_as_judge(truth, model_answer)
    f1_llm_combined = (f1 * 0.5) + (llm_judge_score * 0.5) # Weighted average

    return {
        "f1": round(f1, 3), "precision": round(precision, 3), "recall": round(recall, 3),
        "cosine": round(float(cosine), 3), "rougeL": round(rougeL, 3),
        "f1_llm_combined": round(f1_llm_combined, 3)
    }
