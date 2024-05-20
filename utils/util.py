
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import pandas as pd


def extract_details(df: pd.DataFrame) -> list:
    """
    Extracts detailed information about question and answer pairs from a DataFrame
    with a high degree of textual similarity based on their TF-IDF vectorized representation.
    
    This function computes the cosine similarity between all combinations of text entries
    in the DataFrame, identifies pairs with a similarity score above 0.9, samples up to 5 such
    pairs randomly, and returns a detailed list of these pairs including their similarity score
    and the text of the questions and answers.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing columns '질문_1', '질문_2', '답변_1', '답변_2', '답변_3', '답변_4', '답변_5',
                         where each '질문' and '답변' represent questions and answers in textual form.

    Returns:
    - list: A list of dictionaries, each dictionary containing details of one pair of highly similar text entries.
            Each dictionary includes keys for '유사도' (similarity score), '질문1_1', '질문1_2', '질문2_1', '질문2_2',
            and '답변1_x', '답변2_x' for x ranging from 1 to 5, representing the questions and answers of each pair.
    
    Example usage:
    >>> df = pd.DataFrame({
    ...     '질문_1': ["What is AI?", "What is AI?"],
    ...     '질문_2': ["Explain AI.", "Describe AI."],
    ...     '답변_1': ["AI is...", "AI refers to..."],
    ...     '답변_2': ["Artificial Intelligence...", "A branch of computer science..."],
    ...     '답변_3': ["It deals with...", "It involves..."],
    ...     '답변_4': ["Its applications...", "Uses include..."],
    ...     '답변_5': ["Examples of AI...", "AI examples include..."]
    ... })
    >>> extract_details(df)
    """
    vectorizer = TfidfVectorizer()
    
    columns_to_combine = ['질문_1', '질문_2', '답변_1', '답변_2', '답변_3', '답변_4', '답변_5']
    combined_text = df[columns_to_combine].agg(' '.join, axis=1)
    tfidf_matrix = vectorizer.fit_transform(combined_text)
    
    # 코사인 유사도 계산
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 유사도가 높은 인덱스 추출 및 중복 쌍 제거
    high_similarity_indices = np.where(cosine_sim > 0.9)
    pairs = np.column_stack(high_similarity_indices)
    # 랜덤 샘플링
    sampled_pairs = pairs[np.random.choice(pairs.shape[0], size=min(5, pairs.shape[0]), replace=False)]
    
    # 상세 데이터 추출
    detailed_pairs = []
    for index1, index2 in sampled_pairs:
        sim = cosine_sim[index1, index2]
        row1 = df.iloc[index1][columns_to_combine]
        row2 = df.iloc[index2][columns_to_combine]
        
        pair_details = {
            '유사도': sim,
            '질문1_1': row1['질문_1'], '질문1_2': row1['질문_2'],
            '질문2_1': row2['질문_1'], '질문2_2': row2['질문_2'],
        }
        
        for idx in range(1, 6):
            pair_details[f"답변1_{idx}"] = row1[f"답변_{idx}"]
            pair_details[f"답변2_{idx}"] = row2[f"답변_{idx}"]
            
        detailed_pairs.append(pair_details)
        
    return detailed_pairs