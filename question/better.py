from sense2vec import Sense2Vec
s2v = Sense2Vec().from_disk('/mnt/c/Users/sanatan/OneDrive/Desktop/Question-MCQ-_Answer_Generation-main/UI/MCQ/question/s2v_old')

from sentence_transformers import SentenceTransformer
model_= SentenceTransformer('all-MiniLM-L12-v2')

from typing import List, Tuple
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_answer_and_distractor_embeddings(answer,candidate_distractors):
  answer_embedding = model_.encode([answer])
  distractor_embeddings = model_.encode(candidate_distractors)
  return answer_embedding,distractor_embeddings

def mmr(doc_embedding: np.ndarray,
        word_embeddings: np.ndarray,
        words: List[str],
        top_n: int = 5,
        diversity: float = 0.9) -> List[Tuple[str, float]]:
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [(words[idx], round(float(word_doc_similarity.reshape(1, -1)[0][idx]), 4)) for idx in keywords_idx]

def get_distract(keywords):
    dis = []
    for originalword in keywords:
        word = originalword.lower()

        word = word.replace(" ", "_")
        sense = s2v.get_best_sense(word)
        most_similar = s2v.most_similar(sense, n=10)

        distractors = []

        for each_word in most_similar:
            append_word = each_word[0].split("|")[0].replace("_", " ")
            if append_word not in distractors and append_word != originalword:
                distractors.append(append_word)
        answer_embedd, distractor_embedds = get_answer_and_distractor_embeddings(originalword,distractors)
        final_distractors = mmr(answer_embedd,distractor_embedds,distractors,4)
        filtered_distractors = []
        for dist in final_distractors:
            filtered_distractors.append (dist[0])
        Filtered_Distractors =  filtered_distractors[1:]
        dis.append(Filtered_Distractors)
    return(dis)

print(get_distract('chloroplasts'))