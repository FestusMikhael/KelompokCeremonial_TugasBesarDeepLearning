# Face Recognition System: FaceNet vs. Vision Transformer (ViT)

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Face%20Recognition-blue)
![Python](https://img.shields.io/badge/Python-3.x-yellow)

Repositori ini berisi implementasi sistem **Face Recognition End-to-End** yang membandingkan dua arsitektur Deep Learning modern: **Convolutional Neural Network (CNN)** menggunakan FaceNet, dan **Transformer** menggunakan Vision Transformer (ViT). Proyek ini bertujuan untuk mengevaluasi kinerja kedua model pada dataset wajah dengan sampel terbatas (*Few-Shot Learning*).

---

## 📂 Dataset & Preprocessing

Dataset dikumpulkan secara mandiri dari mahasiswa kelas, dengan tantangan ketersediaan data yang minim (hanya **4 foto per identitas**).

* **Total Kelas:** 70 Mahasiswa.
* **Preprocessing Pipeline:**  
  * **Konversi Format:** Penanganan format `.HEIC` / `.WebP` ke `.JPG`.  
  * **Face Detection:** Menggunakan **MTCNN** untuk mendeteksi dan *crop* wajah secara otomatis.  
  * **Splitting:** Menggunakan strategi **Custom Split** (1 foto per siswa untuk Test, sisanya untuk Train) untuk menjamin setiap identitas ada di data uji.  

---

## 🧠 Arsitektur Model & Metode Training

### 1. FaceNet (InceptionResnetV1)
Model berbasis CNN yang dirancang khusus untuk pengenalan wajah.

* **Backbone:** InceptionResnetV1
* **Pre-trained Weights:** `vggface2` (Dataset wajah skala besar)
* **Input Size:** 160x160 px
* **Loss Function:** `CrossEntropyLoss`
* **Strategi Training (2-Stage):**
    * **Stage 1 (Freeze):** Membekukan backbone dan hanya melatih *classifier head*.
    * **Stage 2 (Fine-Tuning):** *Unfreeze* blok terakhir InceptionResnet dengan *Learning Rate* kecil untuk adaptasi fitur.

### 2. Vision Transformer (ViT Base)
Model berbasis *Self-Attention* yang memproses gambar sebagai urutan *patch*.

* **Backbone:** `vit_base_patch16_224`
* **Pre-trained Weights:** `ImageNet` (Objek umum)
* **Input Size:** 224x224 px
* **Loss Function:** **ArcFace Loss** (Additive Angular Margin Loss) untuk memaksimalkan jarak antar kelas.
* **Augmentasi Data:** Menggunakan augmentasi agresif (`ColorJitter`, `RandomRotation`, `Affine`) untuk mencegah *overfitting*.
* **Strategi Training (2-Stage):**
    * **Stage 1 (Warmup):** *Hard Freeze* pada Body ViT, melatih ArcFace Head selama 40 epoch.
    * **Stage 2 (Full Fine-Tune):** *Unfreeze* seluruh model dengan LR sangat kecil (`1e-7`) untuk Body dan LR standar (`1e-5`) untuk Head.

---

## 📊 Hasil Eksperimen

<img width="1000" height="1009" alt="image" src="https://github.com/user-attachments/assets/d6f1da5d-acc7-41cc-8ae8-259c54469a29" /> <img width="1000" height="1000" alt="image" src="https://github.com/user-attachments/assets/6ce68fe7-cc0e-4089-abc7-0b36c5a030ad" />

| Model | Akurasi | F1-Score (Weighted) | Kesimpulan |
| :--- | :---: | :---: | :--- |
| **FaceNet** | **93.0%** | **0.91** | ✅ Efektif |
| ViT Base | 66.0% | 0.59 | *Underperforming* |

Hasil eksperimen menunjukkan disparitas performa yang signifikan antara kedua arsitektur model pada data uji. Model berbasis CNN, **FaceNet (InceptionResnetV1)**, terbukti menjadi model yang paling efektif dengan mencatatkan Akurasi **93.0%** dan F1-Score (Weighted) **0.91**, yang mengindikasikan keseimbangan tinggi antara presisi dan recall.

Sebaliknya, model **Vision Transformer (ViT)** mengalami kesulitan dalam melakukan generalisasi pada dataset terbatas, hanya mampu mencapai akurasi **66.0%** dengan F1-Score **0.59**. Perbedaan kualitas ini divisualisasikan secara jelas melalui *Confusion Matrix*, di mana FaceNet menampilkan konsentrasi prediksi yang kuat pada garis diagonal (prediksi benar), sementara ViT menunjukkan pola prediksi yang tersebar (*noise*) yang menandakan tingginya tingkat kesalahan klasifikasi antar identitas.

---

## 💡 Analisis

1.  **Pre-training Domain (Wajah vs Objek)**
    * **FaceNet** menggunakan bobot `vggface2`, yang berarti model sudah pernah melihat jutaan wajah manusia sebelumnya. Model sudah memahami fitur spesifik seperti bentuk mata, hidung, dan struktur rahang.
    * **ViT** menggunakan bobot `ImageNet`, yang berisi objek umum (kucing, mobil, buah). ViT harus belajar mengenali fitur wajah dari nol hanya dengan dataset kecil, yang sangat sulit dilakukan.

2.  **Inductive Bias & Ukuran Data**
    * **CNN (FaceNet)** memiliki *Inductive Bias* yang kuat terhadap spasial (piksel berdekatan saling berhubungan). Ini membuat CNN sangat efisien belajar dari data sedikit (*data efficient*).
    * **Transformer (ViT)** tidak memiliki bias spasial bawaan (melihat hubungan global antar patch). ViT dikenal *"Data Hungry"*; biasanya membutuhkan ribuan hingga jutaan gambar per kelas untuk bekerja optimal. Dengan hanya 4 foto per siswa, ViT gagal melakukan generalisasi (gagal menangkap pola wajah yang konsisten).

3.  **Kompleksitas Loss Function**
    * Meskipun ViT menggunakan **ArcFace** (yang secara teori lebih canggih dari CrossEntropy), ArcFace sangat sulit konvergen jika *embeddings* dasar dari backbone belum cukup matang. Karena backbone ViT kesulitan mengekstrak fitur wajah (akibat poin 1 & 2), ArcFace tidak dapat bekerja maksimal memisahkan antar identitas.

---

## 👥 Kelompok Ceremonial

1.  **Festus Mikhael** (122140087)
2.  **Kenneth Austin Wijaya** (122140043)
3.  **Garland Wijaya** (122140001)
