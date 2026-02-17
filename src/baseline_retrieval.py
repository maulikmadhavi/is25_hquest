import numpy as np
import scipy.spatial.distance as dist
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import math

# Try to import the C++ DTW implementation, fall back to Python if not available
try:
    import dtw_cpp

    _dtw_computer = dtw_cpp.DTWComputer()
    USE_CPP_DTW = True
except ImportError:
    USE_CPP_DTW = False
    print("[WARNING] C++ DTW module not available. Using Python implementation.")


def retrieve_tfidf(query_tokens, vectorizer, tfidf_matrix, audio_filenames, top_k=10):
    query_str = ",".join(query_tokens)
    query_vector = vectorizer.transform([query_str])
    scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [(audio_filenames[i], scores[i]) for i in top_indices]


def compute_dtw_distance_python(query_tokens, audio_tokens):
    """Python fallback implementation of DTW"""
    dist_mat = np.array([[dist.euclidean([q], [a]) for a in audio_tokens] for q in query_tokens])
    N, M = dist_mat.shape
    cost_mat = np.zeros((N + 1, M + 1))
    cost_mat[1:, 0] = np.inf
    cost_mat[0, 1:] = np.inf
    for i in range(N):
        for j in range(M):
            penalties = [cost_mat[i, j], cost_mat[i, j + 1], cost_mat[i + 1, j]]
            move = np.argmin(penalties)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalties[move]
    return cost_mat[N, M]


def compute_dtw_distance(query_tokens, audio_tokens):
    """Compute DTW distance using C++ if available, otherwise Python"""
    if not USE_CPP_DTW:
        return compute_dtw_distance_python(query_tokens, audio_tokens)

    # Convert to numpy arrays for C++ implementation
    query_array = np.array(query_tokens, dtype=np.float64)
    audio_array = np.array(audio_tokens, dtype=np.float64)

    return _dtw_computer.compute_dtw_distance(query_array, audio_array)


def retrieve_dtw(query_tokens, audio_sequences, audio_filenames, top_k=10):
    query_tokens = [int(t) for t in query_tokens if t]
    if not query_tokens:
        return [(audio_filenames[0], float("inf"))] * top_k

    distances = []

    # If C++ implementation is available, use batch computation for efficiency
    if USE_CPP_DTW:
        query_array = np.array(query_tokens, dtype=np.float64)
        audio_arrays = []
        valid_indices = []

        for i, seq in enumerate(audio_sequences):
            audio_tokens = [int(t) for t in seq.split(",") if t]
            if audio_tokens:
                audio_arrays.append(np.array(audio_tokens, dtype=np.float64))
                valid_indices.append(i)
            else:
                distances.append((float("inf"), i))

        # Batch compute DTW distances
        if audio_arrays:
            dtw_distances = _dtw_computer.compute_dtw_batch(query_array, audio_arrays)
            for idx, dtw_dist in enumerate(dtw_distances):
                distances.append((dtw_dist, valid_indices[idx]))
    else:
        # Python fallback
        for i, seq in enumerate(audio_sequences):
            audio_tokens = [int(t) for t in seq.split(",") if t]
            if not audio_tokens:
                distances.append((float("inf"), i))
                continue
            dtw_distance = compute_dtw_distance(query_tokens, audio_tokens)
            distances.append((dtw_distance, i))

    distances.sort()
    return [(audio_filenames[i], d) for d, i in distances[:top_k]]


class BigTableInvertedIndex:
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.doc_freq = {}
        self.N = 0
        self.audio_filenames = []

    def build_index(self, audio_sequences, audio_filenames):
        self.N = len(audio_sequences)
        self.audio_filenames = audio_filenames
        token_doc_set = defaultdict(set)
        for idx, seq_str in enumerate(audio_sequences):
            tokens = [t for t in seq_str.split(",") if t]
            freqs = Counter(tokens)
            for token, freq in freqs.items():
                self.inverted_index[token].append((idx, freq))
                token_doc_set[token].add(idx)
        self.doc_freq = {token: len(doc_set) for token, doc_set in token_doc_set.items()}

    def compute_idf(self, token):
        df = self.doc_freq.get(token, 0)
        return math.log((self.N + 1) / (df + 1)) + 1

    def retrieve(self, query_tokens, top_k=10):
        scores = defaultdict(float)
        token_freq_in_query = Counter(query_tokens)
        for token in set(query_tokens):
            postings = self.inverted_index.get(token, [])
            idf = self.compute_idf(token)
            query_tf = token_freq_in_query[token]
            for audio_idx, audio_tf in postings:
                scores[audio_idx] += query_tf * idf * audio_tf
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self.audio_filenames[idx], score) for idx, score in ranked[:top_k]]
