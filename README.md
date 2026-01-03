# 📰 MIND Content-Based News Recommender (Streamlit)

Sistem rekomendasi berita berbasis **Content-Based Filtering** menggunakan  
**TF-IDF + Cosine Similarity** dengan dataset **MIND-small**.

Aplikasi dilengkapi **GUI interaktif berbasis Streamlit** yang ramah untuk
pengguna awam namun tetap informatif untuk kebutuhan akademik.

---

## 🎯 Tujuan Project
Project ini dikembangkan sebagai **Tugas Besar Sistem Pemberi Rekomendasi** dengan tujuan:
- Mengimplementasikan sistem rekomendasi berbasis konten
- Menyediakan antarmuka pengguna yang mudah digunakan
- Menyediakan evaluasi model dengan metrik klasifikasi umum

---

## ✨ Fitur Utama
- 🔍 Pencarian & pemilihan riwayat bacaan (history)
- 🧠 Content-Based Recommendation (TF-IDF + Cosine Similarity)
- 📰 Tampilan hasil rekomendasi:
  - Kartu (mudah dibaca)
  - Tabel (ringkas)
- 📈 Skor relevansi per item
- 🧩 Explainability opsional (kata kunci TF-IDF)
- ♻️ Tombol reset hasil rekomendasi
- 📊 Evaluasi model:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - AUC

---

## 🗂️ Struktur Folder Project
```
.
├── app/
│   ├── streamlit_app.py
│   ├── recommender.py
│   ├── artifacts_loader.py
│   └── ui_components.py
│
├── scripts/
│   └── convert_mind_tsv_to_csv.py
│
├── notebooks/
│   ├── content_based_classification.ipynb
│   └── artifacts_classification_v2/   (generated, not committed)
│
├── datasets/
│   ├── MINDsmall_train/                (downloaded, not committed)
│   └── MINDsmall_dev/                  (downloaded, not committed)
│
├── requirements.txt
├── README.md
└── .gitignore
```

⚠️ **Catatan:**  
Folder `datasets/` dan `artifacts_classification_v2/` **tidak disertakan**
dalam repository ini karena ukuran besar dan ketentuan penggunaan dataset.

---

## 🧰 Requirements
- Python **3.10+** (disarankan)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📥 Download Dataset (MIND-small)
Dataset **MIND-small** dapat diunduh dari sumber resmi Microsoft.

Struktur dataset yang diharapkan:
```
datasets/
├── MINDsmall_train/
│   ├── news.tsv (atau news.csv)
│   └── behaviors.tsv (atau behaviors.csv)
└── MINDsmall_dev/
    ├── news.tsv (atau news.csv)
    └── behaviors.tsv (atau behaviors.csv)
```

Dataset **tidak disertakan** dalam repository ini.

---

## 🔄 (Opsional) Konversi TSV ke CSV
Jika ingin menggunakan versi CSV:
```bash
python scripts/convert_mind_tsv_to_csv.py
```

---

## 🧠 Generate Artifacts (WAJIB)
Sebelum menjalankan aplikasi GUI, artifacts model harus dibuat terlebih dahulu.

1. Buka notebook:
```
notebooks/content_based_classification.ipynb
```

2. Jalankan seluruh cell sampai selesai

Notebook akan menghasilkan:
```
notebooks/artifacts_classification_v2/
├── tfidf_vectorizer.pkl
├── news_all.pkl
├── all2idx.pkl
├── X_all_tfidf.npz
└── metrics.csv
```

---

## ▶️ Menjalankan Aplikasi Streamlit
Setelah dataset dan artifacts tersedia:
```bash
streamlit run app/streamlit_app.py
```

Aplikasi akan terbuka di browser:
```
http://localhost:8501
```

---

## 🧪 Cara Menggunakan Aplikasi
1. Cari berita menggunakan kata kunci (opsional)
2. Pilih beberapa berita sebagai riwayat bacaan
3. Klik **Buat Rekomendasi**
4. Lihat hasil rekomendasi (kartu / tabel)
5. (Opsional) Aktifkan explainability

---

## 📊 Evaluasi Model
Evaluasi dilakukan sebagai klasifikasi biner (relevan vs tidak relevan)
menggunakan metrik:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC

Hasil evaluasi tersimpan di:
```
notebooks/artifacts_classification_v2/metrics.csv
```

---

## 📌 Catatan Akademik
- Sistem ini **tidak menggunakan collaborative filtering**
- Rekomendasi sepenuhnya berbasis kemiripan konten
- Fokus pada interpretabilitas dan kemudahan penggunaan

---

## ⚖️ Lisensi & Disclaimer
- Dataset MIND memiliki **Terms of Use** dari penyedia aslinya
- Repository ini hanya menyertakan **kode dan dokumentasi**
- Pengguna wajib mengunduh dataset dari sumber resmi

---

## 👤 Author
Project ini dikembangkan untuk keperluan akademik  
sebagai Tugas Besar mata kuliah **Sistem Pemberi Rekomendasi**.
