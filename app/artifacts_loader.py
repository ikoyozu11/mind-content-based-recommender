import os
import pickle
import pandas as pd
from scipy import sparse


def resolve_artifact_dir(default_rel: str = "../notebooks/artifacts_classification_v2") -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), default_rel))


def assert_artifacts_exist(artifact_dir: str) -> None:
    required = [
        "tfidf_vectorizer.pkl",
        "news_all.pkl",
        "all2idx.pkl",
        "X_all_tfidf.npz",
        "metrics.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(artifact_dir, f))]
    if missing:
        raise FileNotFoundError(
            "Artifact tidak lengkap.\n"
            f"Folder: {artifact_dir}\n"
            f"Missing: {missing}\n\n"
        )


def load_artifacts(artifact_dir: str):
    """
    Return:
      vectorizer, news_all_df, all2idx_dict, X_all_sparse, metrics_df
    """
    assert_artifacts_exist(artifact_dir)

    with open(os.path.join(artifact_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(artifact_dir, "news_all.pkl"), "rb") as f:
        news_all = pickle.load(f)

    with open(os.path.join(artifact_dir, "all2idx.pkl"), "rb") as f:
        all2idx = pickle.load(f)

    X_all = sparse.load_npz(os.path.join(artifact_dir, "X_all_tfidf.npz"))
    metrics = pd.read_csv(os.path.join(artifact_dir, "metrics.csv"))

    for col in ["news_id", "title", "category", "subcategory"]:
        if col not in news_all.columns:
            pass

    return vectorizer, news_all, all2idx, X_all, metrics
