import os
import numpy as np
import hnswlib
from Bio.Align import PairwiseAligner


def smith_waterman(seq1, seq2):
    seq1_str = " ".join(map(str, seq1))
    seq2_str = " ".join(map(str, seq2))
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 1
    aligner.mismatch_score = 0
    score = aligner.score(seq1_str, seq2_str)
    return score


def build_hnsw_index(tfidf_matrix, dim, index_path="hnsw_index.bin"):
    index = hnswlib.Index(space="cosine", dim=dim)
    if os.path.exists(index_path):
        # Check if the saved index has the correct dimensions
        try:
            test_index = hnswlib.Index(space="cosine", dim=dim)
            test_index.load_index(index_path, max_elements=tfidf_matrix.shape[0])
            if test_index.get_ef() > 0:  # Successfully loaded
                print(f"[INFO] Loading existing HNSW index from {index_path}")
                index = test_index
                return index
        except Exception:
            pass
        # If loading failed or dimensions don't match, rebuild
        print("[INFO] Rebuilding HNSW index (dimension mismatch or corrupted file)")
        os.remove(index_path)

    print(f"[INFO] Creating new HNSW index with {dim} dimensions...")
    index.init_index(max_elements=tfidf_matrix.shape[0], ef_construction=200, M=16)
    index.add_items(tfidf_matrix.toarray())
    index.save_index(index_path)
    print(f"[INFO] HNSW index saved to {index_path}")
    index.set_ef(50)
    return index


def retrieve_hnsw(index, query_tokens, vectorizer, top_k=50):
    query_str = ",".join(query_tokens)
    query_vector = vectorizer.transform([query_str]).toarray()
    labels, distances = index.knn_query(query_vector, k=top_k)
    return labels[0], distances[0]


def H_quest(query_tokens, audio_sequences, audio_filenames, top_indices, top_k=10):
    smith_scores = [smith_waterman(query_tokens, audio_sequences[idx].split(",")) for idx in top_indices]
    sorted_indices = np.argsort(smith_scores)[-top_k:][::-1]
    return [(audio_filenames[top_indices[i]], smith_scores[i]) for i in sorted_indices]
