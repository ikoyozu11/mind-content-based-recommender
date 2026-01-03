# app/streamlit_app.py
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from artifacts_loader import resolve_artifact_dir, load_artifacts
from recommender import (
    recommend_topn,
    build_user_profile,
    explain_top_terms,
    safe_news_meta,
)
from ui_components import render_metrics_cards, render_recs_table

st.set_page_config(
    page_title="MIND Content-Based Recommender",
    page_icon="📰",
    layout="wide"
)

# =========================
# FIXED ARTIFACT PATH (NO UI INPUT)
# =========================
ARTIFACT_DIR = resolve_artifact_dir("../notebooks/artifacts_classification_v2")

@st.cache_resource(show_spinner=True)
def cached_load():
    return load_artifacts(ARTIFACT_DIR)

st.title("📰 Sistem Rekomendasi Berita (Content-Based)")
st.caption("TF-IDF + Cosine Similarity dari riwayat bacaan pengguna (MIND-small)")

try:
    vectorizer, news_all, all2idx, X_all, metrics_df = cached_load()
except Exception as e:
    st.error(
        "Gagal memuat model/data internal.\n\n"
        f"Detail: {e}\n\n"
        "Pastikan folder artifacts ada di:\n"
        f"{ARTIFACT_DIR}"
    )
    st.stop()

# =========================
# SIDEBAR — USER FRIENDLY
# =========================
st.sidebar.header("⚙️ Pengaturan Rekomendasi")

top_n = st.sidebar.slider("Jumlah rekomendasi ditampilkan (Top-N)", 5, 30, 10, 1)

prioritize_recent = st.sidebar.checkbox(
    "Prioritaskan bacaan terbaru",
    value=True,
    help="Jika aktif, riwayat bacaan yang lebih baru akan lebih memengaruhi rekomendasi."
)

candidate_pool = st.sidebar.slider(
    "Jumlah berita yang dipertimbangkan",
    2000, 80000, 20000, 1000,
    help="Semakin besar, rekomendasi bisa lebih beragam tapi proses sedikit lebih lama."
)

more_varied = st.sidebar.checkbox(
    "Rekomendasi lebih bervariasi",
    value=False,
    help="Jika aktif, sistem mengambil kandidat berita secara acak agar hasil tidak monoton."
)

st.sidebar.divider()
show_advanced = st.sidebar.checkbox("Tampilkan mode lanjutan (advanced)", value=False)

if show_advanced:
    seed = st.sidebar.number_input(
        "Angka konsistensi (seed)",
        value=42, step=1,
        help="Agar hasil acak bisa konsisten saat demo. Jika sama, hasil cenderung sama."
    )
else:
    seed = 42

# Default threshold from metrics
default_thr = 0.04
if metrics_df is not None and not metrics_df.empty and "threshold" in metrics_df.columns:
    try:
        default_thr = float(metrics_df.iloc[0]["threshold"])
    except Exception:
        pass

# =========================
# HELPERS UI
# =========================
def badge_label(score: float, thr: float):
    return "🟢 Direkomendasikan" if score >= thr else "⚪ Kurang relevan"

def score_label(score: float):
    # Biar user awam paham: tampilkan skor 0-1 dengan 3 desimal
    return f"{score:.3f}"

def render_score_hist(scores: np.ndarray, thr: float):
    """
    Histogram distribusi skor rekomendasi (Top-N).
    """
    if scores is None or len(scores) == 0:
        st.info("Belum ada skor untuk ditampilkan.")
        return

    fig = plt.figure(figsize=(6, 3.5))
    plt.hist(scores, bins=10)
    plt.axvline(thr, linestyle="--")  # garis threshold
    plt.title("Distribusi Skor Rekomendasi (Top-N)")
    plt.xlabel("Skor relevansi (0–1)")
    plt.ylabel("Jumlah item")
    plt.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.caption(
        "Catatan: Skor relevansi menunjukkan tingkat kemiripan konten dengan minat pengguna "
        "(semakin besar skor, semakin relevan). Garis putus-putus adalah threshold."
    )

def render_recommendation_cards(rec_rows, threshold, show_terms=False, top_explain=5):
    """
    Render rekomendasi dalam bentuk cards.
    rec_rows: list of dict {rank, news_id, score, pred_label, category, subcategory, title}
    """
    if not rec_rows:
        st.info("Belum ada rekomendasi.")
        return

    for r in rec_rows:
        rank = r["rank"]
        nid = r["news_id"]
        title = r.get("title", "")
        category = r.get("category", "")
        subcategory = r.get("subcategory", "")
        score = float(r.get("score", 0.0))

        # Card container
        with st.container(border=True):
            # Header row
            c1, c2 = st.columns([5, 2])
            with c1:
                st.markdown(f"### #{rank} — {title if title else nid}")
                st.caption(f"Kategori: **{category}** | Subkategori: **{subcategory}**")
            with c2:
                st.metric(
                    label="Skor relevansi",
                    value=score_label(score),
                    help="Nilai 0–1 yang menunjukkan tingkat kecocokan konten dengan minat pengguna."
                )
                st.write(badge_label(score, threshold))

            # Optional explainability
            if show_terms:
                with st.expander("Mengapa direkomendasikan? (kata kunci utama)"):
                    uvec = st.session_state.get("uvec_for_explain", None)
                    idx = all2idx.get(nid)
                    if uvec is None or idx is None:
                        st.write("Penjelasan tidak tersedia.")
                    else:
                        item_vec = X_all[idx]
                        terms = explain_top_terms(vectorizer, uvec, item_vec, top_k=10)
                        colL, colR = st.columns(2)
                        with colL:
                            st.write("Kata kunci dari minat pengguna")
                            st.dataframe(pd.DataFrame(terms["user_terms"], columns=["term", "weight"]),
                                         use_container_width=True, hide_index=True)
                        with colR:
                            st.write("Kata kunci dari berita ini")
                            st.dataframe(pd.DataFrame(terms["item_terms"], columns=["term", "weight"]),
                                         use_container_width=True, hide_index=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["🎯 Rekomendasi", "📊 Evaluasi Model", "ℹ️ Tentang"])

# =========================
# TAB 2: EVALUATION
# =========================
with tab2:
    st.subheader("Hasil Evaluasi Model")
    st.write("Metrik evaluasi diambil dari proses evaluasi yang sudah dijalankan sebelumnya (artifacts).")
    render_metrics_cards(metrics_df)

    with st.expander("Lihat tabel lengkap metrics.csv"):
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# =========================
# TAB 3: ABOUT
# =========================
with tab3:
    st.subheader("Tentang Sistem")
    st.markdown(
        """
Sistem ini adalah **Content-Based Recommender** yang memberi rekomendasi berita berdasarkan
kemiripan konten (judul/abstrak) dengan riwayat bacaan pengguna.

**Cara kerja singkat:**
1) Pengguna memilih beberapa berita sebagai *riwayat bacaan*  
2) Sistem membentuk *profil minat* dari riwayat bacaan tersebut  
3) Sistem menghitung kemiripan kandidat berita menggunakan **TF-IDF + cosine similarity**  
4) Sistem menampilkan **Top-N** berita dengan skor kemiripan tertinggi

**Catatan:** Sistem ini tidak menggunakan kemiripan antar pengguna (collaborative). Fokusnya adalah
rekomendasi berbasis konten dan mudah dijelaskan (*explainable*).
        """
    )

# =========================
# TAB 1: RECOMMENDER
# =========================
with tab1:
    st.subheader("Buat Rekomendasi dari Riwayat Bacaan")

    st.info(
        "Petunjuk singkat:\n"
        "1) Cari berita dengan kata kunci (opsional)\n"
        "2) Pilih beberapa berita sebagai riwayat bacaan\n"
        "3) Klik **Buat Rekomendasi** untuk melihat hasil",
        icon="✅"
    )

    # Threshold: show only in advanced
    if show_advanced:
        threshold = st.slider(
            "Ambang batas relevansi (threshold)",
            0.0, 1.0, float(default_thr), 0.01,
            help="Dipakai untuk memberi label 'Direkomendasikan' atau 'Kurang relevan'."
        )
    else:
        threshold = float(default_thr)

    # Prepare helper df for selection
    if "title" in news_all.columns and "news_id" in news_all.columns:
        helper_df = news_all[["news_id", "title", "category", "subcategory"]].copy()
    else:
        helper_df = pd.DataFrame({"news_id": list(all2idx.keys()), "title": [""] * len(all2idx)})

    st.markdown("### 1) Cari & Pilih Riwayat Bacaan")
    keyword = st.text_input(
        "Cari judul berita (kata kunci) — opsional",
        value="",
        placeholder="contoh: sports, covid, election..."
    )

    filtered = helper_df
    if keyword.strip():
        kw = keyword.strip().lower()
        filtered = helper_df[helper_df["title"].astype(str).str.lower().str.contains(kw, na=False)]

    c1, c2 = st.columns([2, 1])
    with c1:
        st.caption(f"Total berita tersedia: {len(helper_df):,} | Hasil pencarian: {len(filtered):,}")
    with c2:
        with st.expander("Lihat hasil pencarian (Top 50)"):
            st.dataframe(filtered.head(50), use_container_width=True, hide_index=True)

    options = filtered["news_id"].head(500).tolist()
    default_pick = options[:5] if len(options) >= 5 else options

    history_ids = st.multiselect(
        "Pilih beberapa berita yang pernah dibaca (disarankan 5–20)",
        options=options,
        default=default_pick
    )

    st.markdown("### 2) Buat Rekomendasi")
    colA, colB, colC = st.columns([1, 1, 2])
    with colA:
        go = st.button("🚀 Buat Rekomendasi", type="primary", use_container_width=True)
    with colC:
        st.caption("Tips: semakin banyak riwayat bacaan dipilih, profil minat lebih akurat.")

    # ========= KUNCI: simpan hasil ke session_state =========
    if go:
        if len(history_ids) < 3:
            st.warning("Pilih minimal 3 berita sebagai riwayat bacaan agar profil minat bisa terbentuk.")
        else:
            with st.spinner("Sedang menghitung rekomendasi..."):
                recs = recommend_topn(
                    history_ids=history_ids,
                    news_all_df=news_all,
                    X_all=X_all,
                    all2idx=all2idx,
                    top_n=top_n,
                    candidate_pool_size=candidate_pool,
                    weighted_profile=prioritize_recent,
                    random_pool=more_varied,
                    seed=int(seed),
                )

            uvec = build_user_profile(history_ids, X_all, all2idx, weighted=prioritize_recent)

            rec_rows = []
            for rank, (nid, score) in enumerate(recs, start=1):
                meta = safe_news_meta(news_all, nid)
                rec_rows.append({
                    "rank": rank,
                    "news_id": nid,
                    "score": float(score),
                    "pred_label": "Direkomendasikan" if score >= threshold else "Kurang relevan",
                    "category": meta.get("category", ""),
                    "subcategory": meta.get("subcategory", ""),
                    "title": meta.get("title", ""),
                })

            # SIMPAN
            st.session_state["rec_rows"] = rec_rows
            st.session_state["uvec_for_explain"] = uvec
            st.session_state["history_ids_last"] = history_ids

            st.success("Rekomendasi berhasil dibuat ✅")

    # ========= tampilkan hasil dari session_state (TIDAK HILANG saat rerun) =========
    rec_rows_saved = st.session_state.get("rec_rows", None)

    if rec_rows_saved:
        st.markdown("### 📰 Hasil Rekomendasi")

        view_mode = st.radio(
            "Tampilan hasil",
            options=["Kartu (lebih mudah dibaca)", "Tabel (ringkas)"],
            horizontal=True,
            key="view_mode_radio"
        )

        show_terms = st.checkbox(
            "Tampilkan penjelasan kata kunci (Explainability)",
            value=False,
            key="show_terms_checkbox",
            help="Jika aktif, tiap rekomendasi bisa dibuka untuk melihat kata kunci utama yang mirip dengan minat pengguna."
        )

        if view_mode.startswith("Kartu"):
            render_recommendation_cards(rec_rows_saved, threshold, show_terms=show_terms)
        else:
            df_out = pd.DataFrame(rec_rows_saved)
            if not show_advanced:
                df_out = df_out.drop(columns=["score"], errors="ignore")
            st.dataframe(df_out, use_container_width=True, hide_index=True)
        
        st.divider()

        col_reset, col_info = st.columns([1, 3])
        with col_reset:
          if st.button("🧹 Reset hasil rekomendasi"):
            st.session_state.pop("rec_rows", None)
            st.session_state.pop("uvec_for_explain", None)
            st.session_state.pop("history_ids_last", None)
            st.rerun()

        with col_info:
          st.caption(
            "Gunakan tombol reset untuk menghapus hasil rekomendasi dan mencoba riwayat bacaan yang berbeda."
          )


        st.download_button(
            "⬇️ Unduh hasil rekomendasi (CSV)",
            data=pd.DataFrame(rec_rows_saved).to_csv(index=False).encode("utf-8"),
            file_name="recommendations.csv",
            mime="text/csv"
        )
