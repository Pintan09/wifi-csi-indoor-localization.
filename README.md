# WiFi CSI Indoor Localization — Kelompok 2
Proyek ini dibuat untuk memenuhi tugas UTS mata kuliah Pembelajaran Mesin Lanjut, dengan topik WiFi CSI Indoor Localization & Positioning.

Penelitian ini menggunakan data Channel State Information (CSI) dari sinyal WiFi untuk mengidentifikasi posisi perangkat di dalam ruangan. Beberapa model pembelajaran mesin digunakan untuk membandingkan kemampuan klasifikasi lokasi berdasarkan karakteristik sinyal CSI.

## Struktur Proyek
- `notebooks/UTS_PML_Kelompok_2.ipynb`
Notebook Google Colab yang berisi pengolahan data, pelatihan model, evaluasi, dan visualisasi hasil.
- `src/preprocessing.py`
Kode untuk membaca dataset dan melakukan preprocessing CSI.
- `src/models.py` 
Arsitektur model CNN yang digunakan pada eksperimen lanjutan.
- `train.py` 
Skrip pelatihan model CNN yang menggunakan fungsi preprocessing dan arsitektur model dari folder src.
- `results/combined/` 
Hasil pengujian berupa model, prediksi, riwayat pelatihan, dan grafik.
- `results/butterworth/`
Checkpoint, riwayat training, dan prediksi CNN amplitude + Butterworth.
- `requirements.txt` 
Daftar library Python yang diperlukan.

## Dataset
Dataset yang digunakan adalah Lab Dataset dari CSI Dataset for Indoor Localization.
Sumber: https://github.com/qiang5love1314/CSI-dataset-for-indoor-localization

Dataset terdiri dari 317 lokasi pengambilan data. Setiap lokasi memiliki data CSI dari 3 antena, 30 subcarrier, dan 1.500 timestep.

## Model yang digunakan 
Model yang dibandingkan dalam penelitian ini meliputi MLP, CNN-LSTM, CNN-only, dan LSTM-only. Selain itu, dilakukan pengujian tambahan menggunakan filter Butterworth dan informasi phase CSI untuk melihat pengaruh preprocessing terhadap performa CNN.

## Metode
- Data CSI kompleks dikonversi menjadi amplitude, dan pada eksperimen gabungan juga menggunakan phase yang disanitasi.
- Amplitude difilter menggunakan low-pass Butterworth orde 4 dengan cutoff ternormalisasi 0,2 pada eksperimen lanjutan.
- Tiap rekaman dibagi menjadi 15 window × 100 timestep.
- Pembagian per lokasi: 10 window training, 2 validation, 3 testing, menghasilkan 3.170/634/951 sampel.
- Normalisasi fitur di-fit hanya pada data training.

## Hasil Pengujian
| Model | Test accuracy | Macro F1 |
|---|---:|---:|
| MLP (amplitude awal) | 92,85% | 0,9148 |
| CNN-LSTM (amplitude awal) | 78,44% | 0,7429 |
| CNN-only (amplitude awal) | **97,06%** | 0,9652 |
| LSTM-only (amplitude awal) | 87,91% | 0,8549 |
| CNN + Butterworth | 96,95% | 0,9635 |
| CNN + Butterworth + sanitized phase | 96,53% | 0,9598 |

Pengujian dilakukan menggunakan 951 sampel data test yang berasal dari 317 lokasi pengambilan data. Berdasarkan hasil pengujian, model CNN-only dengan fitur amplitude awal memperoleh akurasi tertinggi sebesar 97,06%. Penggunaan filter Butterworth dan penambahan fitur phase pada eksperimen ini belum memberikan peningkatan akurasi dibandingkan CNN-only awal.

## Menjalankan
1. Buka file `notebooks/UTS_PML_Kelompok_2.ipynb` menggunakan Google Colab.
2. Aktifkan GPU jika tersedia.
3. Unduh dataset dengan menjalankan perintah berikut pada cell Google Colab:
   ```python
   !git clone https://github.com/qiang5love1314/CSI-dataset-for-indoor-localization.git
   ```
4. Pastikan lokasi dataset sesuai dengan path yang digunakan dalam notebook.
5. Jalankan cell secara berurutan untuk melakukan preprocessing, pelatihan, dan evaluasi model.

Catatan: Eksperimen utama dijalankan melalui notebook Google Colab. File `train.py` disediakan sebagai skrip pelatihan terpisah dan belum diuji secara menyeluruh pada dataset lengkap.

## Keterbatasan 
- Train/validation/test merupakan window berbeda dari rekaman lokasi yang sama. Akurasi ini belum membuktikan generalisasi terhadap sesi atau ruangan baru.
- Galat posisi dinyatakan dalam grid units, bukan meter.
- Perbandingan amplitude awal dan amplitude + phase tidak mengisolasi pengaruh phase saja, karena filtering juga berubah; ablation Butterworth membantu memisahkan sebagian pengaruh tersebut.
- Untuk inferensi checkpoint `.pth` pada data baru, preprocessing dan scaler harus sama dengan yang digunakan ketika training.

