# 🌾 Prediksi Luas Panen Padi di Sumatera dengan Random Forest Regression

Proyek *data mining* untuk memprediksi **Luas Panen padi (hektare)** di wilayah Pulau Sumatera berdasarkan data produksi dan faktor iklim, menggunakan algoritma **Random Forest Regression** yang dioptimalkan dengan **GridSearchCV**.

---

## 📌 Latar Belakang dan Rumusan Masalah

Luas panen merupakan indikator penting dalam perencanaan produksi pangan. Proyek ini menjawab pertanyaan:

> **Bagaimana menentukan Luas Panen dari dataset ini?**

Model dilatih untuk mempelajari hubungan antara tahun, produksi padi, dan kondisi iklim (curah hujan, kelembapan, suhu) terhadap luas panen.

## 📊 Dataset

| Keterangan | Detail |
|---|---|
| File | `Data_Tanaman_Padi_Sumatera_version_1.csv` |
| Cakupan | Provinsi-provinsi di Pulau Sumatera |
| Periode | 1993 – 2020 |
| Jumlah data | 224 baris |

**Kolom pada dataset:**

| Kolom | Peran | Keterangan |
|---|---|---|
| `Provinsi` | Tidak dipakai | Nama provinsi |
| `Tahun` | Fitur | Tahun pencatatan |
| `Produksi` | Fitur | Jumlah produksi padi |
| `Curah hujan` | Fitur | Curah hujan |
| `Kelembapan` | Fitur | Kelembapan udara (%) |
| `Suhu rata-rata` | Fitur | Suhu rata-rata (°C) |
| `Luas Panen` | **Target** | Luas panen padi |

## ⚙️ Alur Pengerjaan

1. **Impor data** dan pengecekan statistik deskriptif (`df.describe()`).
2. **Penentuan fitur dan target**: `Tahun`, `Produksi`, `Curah hujan`, `Kelembapan`, `Suhu rata-rata` → `Luas Panen`.
3. **Pembagian data**: 70% data latih dan 30% data uji (`random_state=42`).
4. **Pemodelan awal**: `RandomForestRegressor` dengan parameter bawaan sebagai *baseline*.
5. **Hyperparameter tuning**: `GridSearchCV` dengan 5-fold cross validation dan skor `R²`.
6. **Evaluasi**: MAE, MSE, dan R² pada data uji, serta plot nilai aktual vs prediksi.
7. **Penyimpanan model**: model terbaik disimpan sebagai `model_rf.pkl` (dan `scaler.pkl`) menggunakan `joblib`.
8. **Prediksi interaktif**: pengguna memasukkan nilai fitur lewat input teks, lalu model menampilkan prediksi luas panen.

### Grid Hyperparameter

| Parameter | Nilai yang Dicoba |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | 10, 20, None |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

## 📈 Hasil Evaluasi

| Model | MAE | MSE | R² |
|---|---|---|---|
| Random Forest (default) | 31.547,50 | 2.680.897.938,34 | 0,9375 |
| **Random Forest (tuned)** | **31.849,15** | **2.463.022.776,86** | **0,9426** |

Setelah *tuning*, R² naik dari **0,9375** menjadi **0,9426** dan MSE turun, sehingga model terbaik mampu menjelaskan sekitar **94,3%** variasi luas panen pada data uji (RMSE ≈ 49.629).

## 🧪 Contoh Prediksi

Berikut contoh hasil dari fitur prediksi interaktif di notebook:

| Tahun | Produksi | Curah Hujan | Kelembapan | Suhu | Prediksi Luas Panen |
|---|---|---|---|---|---|
| 2020 | 1.900.390 | 8.902 | 87 | 34 | 420.476,20 |

## 📁 Struktur Proyek

```
random-forrest-regression-v2/
├── Angga_Fadzar_2306201_RFR.ipynb   # Notebook: pemodelan, tuning, evaluasi, prediksi
├── model_rf.pkl                     # Model Random Forest terbaik
├── scaler.pkl                       # StandardScaler (disimpan dari notebook)
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/angfdzr/random-forrest-regression-v2.git
cd random-forrest-regression-v2
```

### 2. Install dependensi

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

### 3. Siapkan dataset

Letakkan file `Data_Tanaman_Padi_Sumatera_version_1.csv` di folder proyek, lalu ubah path pada sel pembacaan data di notebook (semula `C:/Users/hp/Documents/...`) menjadi lokasi file di komputer Anda.

### 4. Jalankan notebook

```bash
jupyter notebook Angga_Fadzar_2306201_RFR.ipynb
```

### 5. Memakai model yang sudah disimpan

```python
import joblib
import pandas as pd

model = joblib.load("model_rf.pkl")

data = pd.DataFrame({
    'Tahun': [2020],
    'Produksi': [1900390],
    'Curah hujan': [2500],
    'Kelembapan': [85],
    'Suhu rata-rata': [27],
})

print(model.predict(data)[0])
```

> Model dilatih pada data **tanpa** penskalaan, sehingga input ke `model_rf.pkl` tidak perlu di-*scale*.

## 🛠️ Teknologi yang Digunakan

- **Python**
- **pandas** & **NumPy**: pengolahan data
- **scikit-learn**: Random Forest Regressor, GridSearchCV, metrik evaluasi
- **matplotlib** & **seaborn**: visualisasi
- **joblib**: penyimpanan model

## ⚠️ Keterbatasan

- Dataset relatif kecil (224 baris), sehingga hasil evaluasi bisa berubah tergantung pembagian data latih dan uji.
- `Produksi` dipakai sebagai fitur dan berkaitan langsung dengan luas panen, sehingga sangat memengaruhi prediksi.
- Kolom `Provinsi` belum dipakai sebagai fitur, padahal karakteristik antarprovinsi bisa berbeda.
- Random Forest tidak mampu mengekstrapolasi di luar rentang data latih. Input yang tidak realistis (misalnya tahun 2030 atau kelembapan 20%) menghasilkan prediksi yang tidak dapat diandalkan.

## 👤 Penulis

**Angga Fadzar** (NIM 2306201)
Tugas Mata Kuliah Data Mining, Semester 4

---

*Proyek ini dibuat untuk keperluan pembelajaran.*
