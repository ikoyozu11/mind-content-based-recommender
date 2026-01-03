import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def safe_news_meta(news_df, news_id: str) -> dict:
    if "news_id" not in news_df.columns:
        return {}
    row = news_df.loc[news_df["news_id"] == news_id]
    if row.empty:
        return {}
    r = row.iloc[0].to_dict()
    return r


def build_user_profile(
    history_ids,
    X_all,
    all2idx,
    weighted: bool = True
):

    idxs = [all2idx.get(n) for n in history_ids if n in all2idx]
    idxs = [i for i in idxs if i is not None]
    if not idxs:
        return None

    if not weighted:
        u = X_all[idxs].sum(axis=0) / len(idxs)
        return np.asarray(u)

    w = np.linspace(1.0, 2.0, num=len(idxs)).astype(np.float32)
    M = X_all[idxs]
    u = (M.multiply(w[:, None])).sum(axis=0) / w.sum()
    return np.asarray(u)


def score_candidates_batch(uvec, cand_ids, X_all, all2idx):
    scores = np.zeros(len(cand_ids), dtype=np.float32)
    if uvec is None or len(cand_ids) == 0:
        return scores

    idxs = [all2idx.get(n, None) for n in cand_ids]
    valid_pos = [j for j, i in enumerate(idxs) if i is not None]
    if not valid_pos:
        return scores

    cols = [idxs[j] for j in valid_pos]
    M = X_all[cols]
    sim = cosine_similarity(uvec, M)[0]

    for j, s in zip(valid_pos, sim):
        scores[j] = float(s)

    return scores


def recommend_topn(
    history_ids,
    news_all_df,
    X_all,
    all2idx,
    top_n: int = 10,
    candidate_pool_size: int = 20000,
    weighted_profile: bool = True,
    random_pool: bool = False,
    seed: int = 42
):

    seen = set(history_ids)

    all_ids = news_all_df["news_id"].tolist() if "news_id" in news_all_df.columns else list(all2idx.keys())

    if random_pool:
        rng = np.random.default_rng(seed)
        candidates = [nid for nid in all_ids if nid not in seen]
        if len(candidates) > candidate_pool_size:
            pool = rng.choice(candidates, size=candidate_pool_size, replace=False).tolist()
        else:
            pool = candidates
    else:
        pool = []
        for nid in all_ids:
            if nid in seen:
                continue
            pool.append(nid)
            if len(pool) >= candidate_pool_size:
                break

    uvec = build_user_profile(history_ids, X_all, all2idx, weighted=weighted_profile)
    scores = score_candidates_batch(uvec, pool, X_all, all2idx)

    order = np.argsort(-scores)
    top = [(pool[i], float(scores[i])) for i in order[:top_n]]
    return top


def explain_top_terms(vectorizer, uvec, item_vec, top_k=10):
    """
    Explainability sederhana:
      - top terms di user profile dan item (berdasarkan TF-IDF tertinggi)
    """
    if uvec is None:
        return {"user_terms": [], "item_terms": []}

    try:
        feats = vectorizer.get_feature_names_out()
    except Exception:
        feats = None

    u = np.asarray(uvec).ravel()
    i = item_vec.toarray().ravel()

    def top_terms(vec):
        if feats is None:
            return []
        idx = np.argsort(-vec)[:top_k]
        out = []
        for j in idx:
            if vec[j] <= 0:
                break
            out.append((str(feats[j]), float(vec[j])))
        return out

    return {
        "user_terms": top_terms(u),
        "item_terms": top_terms(i),
    }


def most_similar_history_items(uvec, history_ids, X_all, all2idx, top_k=3):
    idxs = [all2idx.get(n) for n in history_ids if n in all2idx]
    idxs = [i for i in idxs if i is not None]
    if not idxs or uvec is None:
        return []

    M = X_all[idxs]
    sims = cosine_similarity(uvec, M)[0]
    order = np.argsort(-sims)[:top_k]
    return [(history_ids[order_pos], float(sims[order_pos])) for order_pos in order]