---
title: "Non-Negative Matrix Factorization (NMF) untuk Topic Modeling"
format:
  revealjs:
    incremental: true   
    code-block-height: 750px
    progress: false
    history: true
    chalkboard: true
---

## Apa itu NMF? Penjelasan Sederhana {.unnumbered}

Bayangkan Anda memiliki **buku resep** dengan 1000 resep dan 500 bahan-bahan.

- **Matrix V** = Tabel resep × bahan (1000 × 500)
- **NMF** memecah tabel ini menjadi 2 tabel kecil:
  - **Matrix W** = Resep × Kategori masakan (1000 × 10)  
  - **Matrix H** = Kategori masakan × Bahan (10 × 500)

$$\underbrace{V}_{1000 \times 500} \approx \underbrace{W}_{1000 \times 10} \times \underbrace{H}_{10 \times 500}$$

**W** menunjukkan seberapa kuat setiap resep termasuk dalam kategori  
**H** menunjukkan bahan-bahan khas untuk setiap kategori

## Mengapa Persamaan V ≈ W × H? {.unnumbered}

Mari kita lihat dengan **contoh kecil**:

Misalnya kita punya 3 dokumen dan 4 kata:

$$V = \begin{bmatrix}
2 & 3 & 0 & 1 \\
1 & 1 & 4 & 2 \\
3 & 2 & 1 & 0
\end{bmatrix}$$

NMF mencari:
$$W = \begin{bmatrix}
0.8 & 0.1 \\
0.2 & 0.9 \\
0.7 & 0.2
\end{bmatrix}, \quad H = \begin{bmatrix}
3 & 4 & 0 & 1 \\
0 & 0 & 4 & 2
\end{bmatrix}$$

Sehingga: $V \approx W \times H$

**Interpretasi:**
- Dokumen 1: 80% topik 1, 10% topik 2  
- Topik 1: kata 1 dan 2 dominan  
- Topik 2: kata 3 dan 4 dominan

## Persamaan Matematika - Step by Step {.unnumbered}

### 1. **Objective Function** (Yang ingin kita minimimalkan):

$$\text{Error} = ||V - WH||^2$$

Artinya: **"Seberapa jauh hasil perkalian WH dari data asli V?"**

### 2. **Dalam bentuk elemen**:
$$\text{Error} = \sum_{i,j} (V_{ij} - (WH)_{ij})^2$$

Dimana $(WH)_{ij} = \sum_{k=1}^{r} W_{ik} \times H_{kj}$

### 3. **Constraint** (Batasan):
Semua nilai dalam W dan H harus ≥ 0

**Mengapa non-negatif?** Karena:
- Frekuensi kata tidak bisa negatif
- Proporsi topik tidak bisa negatif
- Hasil lebih mudah diinterpretasi

## Algoritma Update - Penjelasan Intuitif {.unnumbered}

NMF menggunakan **Multiplicative Update Rules**:

### Update untuk W:
$$W_{ik}^{\text{new}} = W_{ik}^{\text{old}} \times \frac{(VH^T)_{ik}}{(WHH^T)_{ik}}$$

### Update untuk H:
$$H_{kj}^{\text{new}} = H_{kj}^{\text{old}} \times \frac{(W^TV)_{kj}}{(W^TWH)_{kj}}$$

**Intuisi:**
- Jika $(VH^T)_{ik} > (WHH^T)_{ik}$ → $W_{ik}$ naik
- Jika $(VH^T)_{ik} < (WHH^T)_{ik}$ → $W_{ik}$ turun
- Algoritma otomatis menjaga nilai tetap positif!

## Kenapa NMF Bagus untuk Topic Modeling? {.unnumbered}

### 🎯 **Keunggulan untuk Teks:**

1. **Non-negatif** → Cocok dengan frekuensi kata (selalu ≥ 0)
2. **Parts-based** → Setiap topik adalah "campuran kata"
3. **Sparse** → Banyak nilai mendekati 0 (realistis untuk teks)
4. **Interpretable** → Mudah memahami topik yang dihasilkan

### 📊 **Contoh Interpretasi:**
- **Matrix W**: "Dokumen tentang apa saja?"
- **Matrix H**: "Topik terdiri dari kata apa saja?"

**Topik 1**: "teknologi, komputer, software, programming"  
**Topik 2**: "makanan, resep, masak, dapur"

## Implementasi Sederhana - Step by Step {.unnumbered}

### Step 1: Import Libraries
```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

# Contoh dokumen sederhana
documents = [
    "kucing lucu bermain bola",
    "anjing suka berlari di taman", 
    "mobil cepat di jalan raya",
    "motor sport balap cepat",
    "kucing tidur di rumah",
    "anjing berlari mengejar bola"
]

print("Dokumen yang akan dianalisis:")
for i, doc in enumerate(documents):
    print(f"{i+1}. {doc}")
```

### Step 2: Buat Document-Term Matrix
```python
# Buat vectorizer (mengubah teks jadi angka)
vectorizer = CountVectorizer(stop_words=None)
X = vectorizer.fit_transform(documents)

print(f"Shape matrix: {X.shape}")  # (6 dokumen, n kata)
print(f"Kata-kata: {vectorizer.get_feature_names_out()}")

# Lihat matrix dalam bentuk array
X_dense = X.toarray()
df = pd.DataFrame(X_dense, 
                  columns=vectorizer.get_feature_names_out(),
                  index=[f"Doc{i+1}" for i in range(len(documents))])
print("\nDocument-Term Matrix:")
print(df)
```

## Implementasi NMF - Mari Kita Coba! {.unnumbered}

### Step 3: Terapkan NMF
```python
# Terapkan NMF dengan 2 topik
n_topics = 2
nmf = NMF(n_components=n_topics, random_state=42, max_iter=100)

# Fit dan transform
W = nmf.fit_transform(X)  # Dokumen × Topik
H = nmf.components_       # Topik × Kata

print(f"Shape W (Document-Topic): {W.shape}")
print(f"Shape H (Topic-Word): {H.shape}")

# Lihat matrix W (seberapa kuat setiap dokumen di setiap topik)
w_df = pd.DataFrame(W, 
                    columns=[f"Topic_{i+1}" for i in range(n_topics)],
                    index=[f"Doc_{i+1}" for i in range(len(documents))])
print("\nMatrix W (Document-Topic Distribution):")
print(w_df.round(3))
```

### Step 4: Interpretasi Topik
```python
def show_topics(model, feature_names, n_top_words=5):
    """Tampilkan kata-kata teratas untuk setiap topik"""
    
    for topic_idx, topic in enumerate(model.components_):
        # Ambil index kata dengan nilai tertinggi
        top_words_idx = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        top_weights = topic[top_words_idx]
        
        print(f"\n📝 Topik {topic_idx + 1}:")
        for word, weight in zip(top_words, top_weights):
            print(f"  - {word}: {weight:.3f}")

feature_names = vectorizer.get_feature_names_out()
show_topics(nmf, feature_names, n_top_words=3)
```

## Visualisasi Hasil - Membuat Grafik {.unnumbered}

```python
# Visualisasi Document-Topic Distribution
plt.figure(figsize=(12, 5))

# Plot 1: Document-Topic Matrix
plt.subplot(1, 2, 1)
plt.imshow(W.T, cmap='Blues', aspect='auto')
plt.colorbar()
plt.title('Document-Topic Distribution')
plt.xlabel('Documents')
plt.ylabel('Topics')
plt.yticks(range(n_topics), [f'Topic {i+1}' for i in range(n_topics)])
plt.xticks(range(len(documents)), [f'Doc {i+1}' for i in range(len(documents))])

# Plot 2: Topic-Word Matrix (hanya beberapa kata teratas)
plt.subplot(1, 2, 2)
# Ambil 8 kata dengan bobot tertinggi untuk visualisasi
top_words_per_topic = []
for topic_idx in range(n_topics):
    top_idx = H[topic_idx].argsort()[-8:][::-1]
    top_words_per_topic.extend(top_idx)

unique_top_words = list(set(top_words_per_topic))
H_subset = H[:, unique_top_words]
word_labels = [feature_names[i] for i in unique_top_words]

plt.imshow(H_subset, cmap='Reds', aspect='auto')
plt.colorbar()
plt.title('Topic-Word Distribution')
plt.xlabel('Words')
plt.ylabel('Topics')
plt.yticks(range(n_topics), [f'Topic {i+1}' for i in range(n_topics)])
plt.xticks(range(len(word_labels)), word_labels, rotation=45)

plt.tight_layout()
plt.show()
```

## Contoh Real: Analisis Berita {.unnumbered}

```python
# Dataset berita yang lebih realistis
news_articles = [
    "Presiden mengumumkan kebijakan ekonomi baru untuk meningkatkan pertumbuhan",
    "Tim nasional sepak bola berhasil menang melawan lawan yang kuat",
    "Teknologi artificial intelligence semakin berkembang pesat di Indonesia", 
    "Bursa saham mengalami kenaikan signifikan hari ini",
    "Pemain bintang mencetak gol spektakuler di pertandingan final",
    "Startup teknologi Indonesia raih pendanaan besar dari investor",
    "Menteri keuangan bahas rencana anggaran tahun depan",
    "Turnamen olahraga internasional akan diselenggarakan di Jakarta",
    "Platform digital baru diluncurkan untuk membantu UMKM",
    "Indeks harga saham gabungan ditutup menguat 2 persen"
]

# Preprocessing yang lebih baik
from sklearn.feature_extraction.text import TfidfVectorizer

# Gunakan TF-IDF untuk hasil yang lebih baik
vectorizer = TfidfVectorizer(
    max_features=50,        # Ambil 50 kata paling penting
    stop_words=None,        # Bisa tambah stopwords bahasa Indonesia
    lowercase=True,
    ngram_range=(1, 1)      # Hanya unigram
)

X = vectorizer.fit_transform(news_articles)
print(f"Ukuran matrix: {X.shape}")

# NMF dengan 3 topik
n_topics = 3
nmf = NMF(n_components=n_topics, 
          random_state=42, 
          alpha=0.1,          # Sedikit regularisasi
          l1_ratio=0.5,       # Mix L1 dan L2
          max_iter=200)

W = nmf.fit_transform(X)
H = nmf.components_

print("\n🔍 HASIL TOPIC MODELING:")
show_topics(nmf, vectorizer.get_feature_names_out(), n_top_words=5)
```

## Interpretasi dan Evaluasi {.unnumbered}

### Interpretasi Document-Topic:
```python
# Lihat dokumen mana yang masuk topik mana
w_df = pd.DataFrame(W, 
                    columns=[f"Topik_{i+1}" for i in range(n_topics)],
                    index=[f"Artikel_{i+1}" for i in range(len(news_articles))])

print("\n📊 DISTRIBUSI TOPIK PER DOKUMEN:")
print(w_df.round(3))

# Tentukan topik dominan untuk setiap dokumen
dominant_topics = np.argmax(W, axis=1)
print("\n🎯 TOPIK DOMINAN:")
for i, topic in enumerate(dominant_topics):
    print(f"Artikel {i+1}: Topik {topic+1}")
    print(f"  '{news_articles[i][:50]}...'")
```

### Evaluasi Kualitas:
```python
def evaluate_nmf_quality(X, W, H):
    """Evaluasi kualitas hasil NMF"""
    
    # 1. Reconstruction Error
    X_reconstructed = W @ H
    if hasattr(X, 'toarray'):  # Jika sparse matrix
        X_dense = X.toarray()
    else:
        X_dense = X
        
    reconstruction_error = np.linalg.norm(X_dense - X_reconstructed, 'fro')
    relative_error = reconstruction_error / np.linalg.norm(X_dense, 'fro')
    
    # 2. Sparsity (seberapa banyak nilai mendekati 0)
    sparsity_W = np.mean(W < 0.01)
    sparsity_H = np.mean(H < 0.01)
    
    # 3. Topic Coherence (sederhana: rata-rata bobot kata teratas)
    coherence_scores = []
    for topic_idx in range(H.shape[0]):
        top_words_weights = np.sort(H[topic_idx])[-5:]  # 5 kata teratas
        coherence_scores.append(np.mean(top_words_weights))
    
    return {
        'reconstruction_error': reconstruction_error,
        'relative_error': relative_error,
        'sparsity_W': sparsity_W,
        'sparsity_H': sparsity_H,
        'avg_coherence': np.mean(coherence_scores)
    }

metrics = evaluate_nmf_quality(X, W, H)
print("\n📈 EVALUASI KUALITAS NMF:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
```

## Parameter Tuning - Mencari Hasil Terbaik {.unnumbered}

### 1. Mencari Jumlah Topik Optimal:
```python
def find_optimal_topics(X, max_topics=10):
    """Cari jumlah topik optimal berdasarkan reconstruction error"""
    
    errors = []
    coherence_scores = []
    topic_range = range(2, max_topics + 1)
    
    for n_topics in topic_range:
        nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
        W = nmf.fit_transform(X)
        H = nmf.components_
        
        # Hitung error
        metrics = evaluate_nmf_quality(X, W, H)
        errors.append(metrics['relative_error'])
        coherence_scores.append(metrics['avg_coherence'])
    
    # Plot hasil
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Reconstruction Error
    ax1.plot(topic_range, errors, 'bo-')
    ax1.set_xlabel('Jumlah Topik')
    ax1.set_ylabel('Relative Reconstruction Error')
    ax1.set_title('Elbow Method: Reconstruction Error')
    ax1.grid(True)
    
    # Plot 2: Coherence Score
    ax2.plot(topic_range, coherence_scores, 'ro-')
    ax2.set_xlabel('Jumlah Topik')
    ax2.set_ylabel('Average Coherence Score')
    ax2.set_title('Topic Coherence vs Number of Topics')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return topic_range, errors, coherence_scores

# Cari jumlah topik optimal
topic_range, errors, coherence_scores = find_optimal_topics(X, max_topics=8)
```

### 2. Hyperparameter Tuning:
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Parameter grid untuk tuning
param_grid = {
    'alpha': [0.0, 0.1, 0.5],
    'l1_ratio': [0.0, 0.5, 1.0],
    'max_iter': [100, 200, 500]
}

best_reconstruction_error = float('inf')
best_params = {}

print("🔧 MENCARI PARAMETER TERBAIK...")
for alpha in param_grid['alpha']:
    for l1_ratio in param_grid['l1_ratio']:
        for max_iter in param_grid['max_iter']:
            
            nmf = NMF(n_components=3, 
                     alpha=alpha,
                     l1_ratio=l1_ratio, 
                     max_iter=max_iter,
                     random_state=42)
            
            try:
                W = nmf.fit_transform(X)
                H = nmf.components_
                
                metrics = evaluate_nmf_quality(X, W, H)
                error = metrics['relative_error']
                
                if error < best_reconstruction_error:
                    best_reconstruction_error = error
                    best_params = {
                        'alpha': alpha,
                        'l1_ratio': l1_ratio, 
                        'max_iter': max_iter,
                        'error': error
                    }
                    
            except:
                continue

print(f"\n✅ PARAMETER TERBAIK: {best_params}")
```

## Studi Kasus Lengkap: Analisis Review Produk {.unnumbered}

```python
# Simulasi review produk
reviews = [
    "produk bagus kualitas sangat memuaskan pelayanan cepat",
    "harga mahal tapi kualitas sebanding recommended", 
    "pengiriman lambat packaging kurang rapi kecewa",
    "produk sesuai deskripsi kualitas oke harga terjangkau",
    "pelayanan ramah fast response penjual recommended",
    "barang rusak saat sampai packaging buruk mengecewakan",
    "kualitas premium harga worth it sangat puas",
    "pengiriman super cepat packaging rapi aman",
    "produk tidak sesuai gambar kualitas mengecewakan", 
    "pelayanan excellent respon cepat sangat membantu"
]

print("🛍️ ANALISIS REVIEW PRODUK E-COMMERCE")
print("=" * 50)

# Preprocessing
vectorizer = TfidfVectorizer(
    max_features=30,
    ngram_range=(1, 2),  # Unigram dan bigram
    lowercase=True
)

X = vectorizer.fit_transform(reviews)

# NMF dengan parameter optimal
nmf = NMF(n_components=3, 
          alpha=0.1, 
          l1_ratio=0.5, 
          max_iter=200,
          random_state=42)

W = nmf.fit_transform(X)
H = nmf.components_

# Tampilkan hasil
print("\n🏷️ TOPIK YANG DITEMUKAN:")
topic_labels = []
for topic_idx, topic in enumerate(H):
    top_words_idx = topic.argsort()[-5:][::-1]
    top_words = [vectorizer.get_feature_names_out()[i] for i in top_words_idx]
    weights = topic[top_words_idx]
    
    print(f"\nTopik {topic_idx + 1}:")
    for word, weight in zip(top_words, weights):
        print(f"  • {word}: {weight:.3f}")
    
    # Label topik berdasarkan kata-kata dominan
    if any(word in ['bagus', 'kualitas', 'memuaskan', 'premium'] for word in top_words):
        topic_labels.append("Kualitas Produk")
    elif any(word in ['pengiriman', 'cepat', 'lambat', 'packaging'] for word in top_words):
        topic_labels.append("Pengiriman & Packaging") 
    elif any(word in ['pelayanan', 'ramah', 'respon'] for word in top_words):
        topic_labels.append("Pelayanan")
    else:
        topic_labels.append(f"Topik {topic_idx + 1}")

print(f"\n🏷️ LABEL TOPIK: {topic_labels}")

# Klasifikasi review
print("\n📝 KLASIFIKASI REVIEW:")
for i, review in enumerate(reviews):
    dominant_topic = np.argmax(W[i])
    confidence = W[i, dominant_topic]
    
    print(f"\nReview {i+1}: {topic_labels[dominant_topic]} (confidence: {confidence:.3f})")
    print(f"  '{review}'")
```

## Tips Praktis untuk Topic Modeling {.unnumbered}

### ✅ **Do's (Yang Harus Dilakukan):**

```python
# 1. Preprocessing yang baik
def preprocess_text_for_nmf(texts):
    """Preprocessing teks untuk NMF"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Gunakan TF-IDF, bukan count
    vectorizer = TfidfVectorizer(
        max_df=0.8,          # Buang kata yang terlalu umum
        min_df=2,            # Buang kata yang terlalu jarang  
        max_features=1000,   # Batasi jumlah fitur
        ngram_range=(1, 2),  # Unigram + bigram
        lowercase=True,
        stop_words='english' # Atau buat stopwords bahasa Indonesia
    )
    
    return vectorizer.fit_transform(texts), vectorizer

# 2. Validasi dengan coherence score
def calculate_topic_coherence(H, vectorizer, top_n=10):
    """Hitung coherence score untuk evaluasi topik"""
    feature_names = vectorizer.get_feature_names_out()
    coherence_scores = []
    
    for topic_idx in range(H.shape[0]):
        # Ambil top words
        top_words_idx = H[topic_idx].argsort()[-top_n:][::-1]
        top_words_weights = H[topic_idx][top_words_idx]
        
        # Simple coherence: rata-rata weight dari top words
        coherence = np.mean(top_words_weights)
        coherence_scores.append(coherence)
    
    return np.mean(coherence_scores)
```

### ❌ **Don'ts (Yang Harus Dihindari):**

- Jangan gunakan terlalu banyak topik untuk data kecil
- Jangan lupa preprocessing (stopwords, normalisasi)
- Jangan abaikan parameter tuning
- Jangan interpretasi topik hanya dari 1-2 kata teratas

## Kesimpulan & Next Steps {.unnumbered}

### 🎯 **Key Takeaways:**

1. **NMF = Pemecahan Matrix** menjadi 2 bagian yang lebih kecil
2. **Non-negatif** = Cocok untuk data teks (frekuensi kata)
3. **W Matrix** = Distribusi topik per dokumen
4. **H Matrix** = Distribusi kata per topik
5. **Interpretable** = Hasil mudah dipahami manusia

### 📚 **Untuk Belajar Lebih Lanjut:**

```python
# Resources yang recommended:
resources = {
    "Paper Asli": "Lee & Seung (2001) - Algorithms for NMF",
    "Library": "scikit-learn NMF documentation", 
    "Dataset Practice": "20 Newsgroups, Reuters-21578",
    "Advanced": "Online NMF, Kernel NMF, Semi-supervised NMF"
}

# Latihan yang bisa dicoba:
exercises = [
    "Analisis tweet tentang topik tertentu",
    "Topic modeling artikel berita Indonesia", 
    "Clustering review produk e-commerce",
    "Ekstraksi topik dari forum diskusi"
]
```

### 🚀 **Challenge untuk Anda:**
Coba implementasikan NMF pada dataset teks pilihan Anda dan bandingkan dengan LDA (Latent Dirichlet Allocation)!

---

## Terima Kasih! 🙏 {.unnumbered}

**Pertanyaan tentang NMF dan Topic Modeling?** 

*Mari diskusi!* 💬

---