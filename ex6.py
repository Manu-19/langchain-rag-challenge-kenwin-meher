"""
TASK 6: Cosine Similarity
---------------------------
Part A: Implement cosine_similarity_manual(v1, v2) WITHOUT
        using numpy.  Use only Python loops / math.
Part B: Implement cosine_similarity_numpy(v1, v2) using numpy.

Both should return a float between -1 and 1.

Then embed these two pairs and print which pair is more similar:
  Pair 1: "dog" vs "puppy"
  Pair 2: "dog" vs "automobile"

Formula:
  cosine_similarity = (v1 · v2) / (||v1|| × ||v2||)

HINT:
  dot product: sum(a*b for a, b in zip(v1, v2))
  magnitude  : sum(x**2 for x in v) ** 0.5
  numpy equiv: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
"""
import math
import numpy as np
from sentence_transformers import SentenceTransformer

def cosine_similarity_manual(v1: list, v2: list) -> float:
    """Computes cosine similarity using pure Python."""
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(x ** 2 for x in v1))
    mag2 = math.sqrt(sum(x ** 2 for x in v2))
    return dot / (mag1 * mag2)


def cosine_similarity_numpy(v1: list, v2: list) -> float:
    """Computes cosine similarity using numpy."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def compare_word_pairs() -> dict:
    """
    Embeds dog/puppy and dog/automobile, returns:
    {
      "dog_vs_puppy"      : float,
      "dog_vs_automobile" : float,
      "more_similar_pair" : str
    }
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    emb_dog = model.encode("dog")
    emb_puppy = model.encode("puppy")
    emb_auto = model.encode("automobile")

    sim1 = cosine_similarity_numpy(emb_dog, emb_puppy)
    sim2 = cosine_similarity_numpy(emb_dog, emb_auto)

    more_similar = "dog_vs_puppy" if sim1 > sim2 else "dog_vs_automobile"

    return {
        "dog_vs_puppy": sim1,
        "dog_vs_automobile": sim2,
        "more_similar_pair": more_similar
    }


result = compare_word_pairs()

print("Dog vs Puppy Similarity:", result["dog_vs_puppy"])
print("Dog vs Automobile Similarity:", result["dog_vs_automobile"])
print("More Similar Pair:", result["more_similar_pair"])