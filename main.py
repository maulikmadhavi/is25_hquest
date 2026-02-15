import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

from src.baseline_retrieval import retrieve_tfidf, retrieve_dtw, BigTableInvertedIndex
from src.hquest_retrieval import build_hnsw_index, retrieve_hnsw, H_quest

def run_retrieval(audio_csv_path, query_csv_path, output_csv_path, top_k=10):
    # Load audio data
    audio_df = pd.read_csv(audio_csv_path)
    audio_filenames = audio_df["Filename"].tolist()
    audio_sequences = audio_df["Data"].tolist()

    # Load query data
    query_df = pd.read_csv(query_csv_path)
    query_filenames = query_df["Filename"].tolist()
    query_sequences = query_df["Data"].tolist()

    # Initialize TF-IDF vectorizer and matrix on audio sequences
    vectorizer = TfidfVectorizer(analyzer=lambda x: x.split(","))
    tfidf_matrix = vectorizer.fit_transform(audio_sequences)

    # Build HNSW index for HQuEST retrieval
    hnsw_index = build_hnsw_index(tfidf_matrix, tfidf_matrix.shape[1], index_path="hnsw_index.bin")

    # Build BigTable inverted index for baseline retrieval
    bigtable_index = BigTableInvertedIndex()
    bigtable_index.build_index(audio_sequences, audio_filenames)

    all_results = []

    for i, query_seq in enumerate(query_sequences):
        query_tokens = query_seq.split(",")

        # Retrieve top-K using TF-IDF cosine similarity
        tfidf_results = retrieve_tfidf(query_tokens, vectorizer, tfidf_matrix, audio_filenames, top_k)

        # Retrieve top-K using DTW distance
        dtw_results = retrieve_dtw(query_tokens, audio_sequences, audio_filenames, top_k)

        # Retrieve top-K using HQuEST (HNSW + Smith-Waterman)
        hnsw_indices, _ = retrieve_hnsw(hnsw_index, query_tokens, vectorizer, top_k)
        hquest_results = H_quest(query_tokens, audio_sequences, audio_filenames, hnsw_indices, top_k)

        # Retrieve top-K using BigTable inverted index TF-IDF scoring
        bigtable_results = bigtable_index.retrieve(query_tokens, top_k)

        # Save all results together
        for j in range(top_k):
            all_results.append([
                query_filenames[i],
                tfidf_results[j][0], tfidf_results[j][1],
                dtw_results[j][0], dtw_results[j][1],
                hquest_results[j][0], hquest_results[j][1],
                bigtable_results[j][0], bigtable_results[j][1]
            ])

    columns = [
        "Query Filename",
        "TF-IDF File", "TF-IDF Score",
        "DTW File", "DTW Distance",
        "HQuEST File", "HQuEST Score",
        "BigTable File", "BigTable Score"
    ]

    df = pd.DataFrame(all_results, columns=columns)
    df.to_csv(output_csv_path, index=False)
    print(f"[✓] Retrieval complete. Results saved to: {output_csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QbE-STD retrieval pipeline with multiple methods.")
    parser.add_argument("--audio_csv", required=True, help="CSV path for target audio token sequences")
    parser.add_argument("--query_csv", required=True, help="CSV path for query audio token sequences")
    parser.add_argument("--output_csv", required=True, help="CSV path to save retrieval results")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top results to retrieve")

    args = parser.parse_args()
    run_retrieval(args.audio_csv, args.query_csv, args.output_csv, args.top_k)
