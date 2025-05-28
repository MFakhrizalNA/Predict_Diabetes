# Laporan Proyek Machine Learning - M. Fakhrizal Nur Athoilah

# Domain Proyek
---
## Latar Belakang
Diabetes merupakan salah satu penyakit kronis yang prevalensinya terus meningkat secara global, termasuk di Indonesia. Deteksi dini terhadap risiko diabetes sangat penting untuk mencegah komplikasi yang lebih serius. Dengan kemajuan teknologi, khususnya dalam bidang machine learning, kini dimungkinkan untuk memprediksi risiko diabetes berdasarkan data medis dan gaya hidup individu.

Dalam proyek ini, digunakan dataset dari Kaggle yang berjudul "Diabetes Prediction Dataset". Dataset ini mencakup berbagai variabel seperti usia, jenis kelamin, tekanan darah, kadar glukosa, dan faktor gaya hidup lainnya. Tujuan dari proyek ini adalah membangun model prediktif yang dapat mengklasifikasikan individu ke dalam kategori berisiko atau tidak berisiko terkena diabetes.

Masalah prediksi diabetes termasuk dalam domain kesehatan masyarakat dan kesehatan digital (e-health), khususnya pada area predictive analytics untuk pencegahan penyakit tidak menular. Diabetes adalah penyakit metabolik kronis yang ditandai dengan kadar gula darah tinggi, yang jika tidak ditangani secara dini, dapat menimbulkan komplikasi serius seperti penyakit jantung, gagal ginjal, kebutaan, bahkan kematian dini.

---
## Mengapa Masalah Ini Perlu Diselesaikan?
Peningkatan kasus diabetes secara global, terutama di negara berkembang seperti Indonesia, menjadi perhatian utama WHO. Menurut data International Diabetes Federation (IDF), pada tahun 2021 terdapat lebih dari 537 juta orang dewasa di dunia yang hidup dengan diabetes, dan angka ini diperkirakan akan meningkat menjadi 643 juta pada tahun 2030 [1]. Di Indonesia sendiri, prevalensi diabetes terus meningkat dari tahun ke tahun menurut data Riskesdas Kemenkes RI [2].

Pendekatan tradisional dalam deteksi dini diabetes umumnya memerlukan pemeriksaan laboratorium yang mahal dan waktu tunggu yang lama. Dengan menerapkan model machine learning yang memanfaatkan data gejala dan demografis, deteksi risiko diabetes dapat dilakukan lebih cepat, murah, dan bersifat preventif. Hal ini penting untuk meningkatkan kualitas hidup dan menekan beban biaya kesehatan nasional.

---
## Bagaimana Masalah Ini Diselesaikan?
Masalah ini diselesaikan dengan membangun sistem prediksi berbasis machine learning menggunakan dataset yang memuat berbagai indikator gejala diabetes dan informasi pasien. Model dikembangkan dengan membandingkan berbagai algoritma klasifikasi, dan hasil terbaik digunakan sebagai dasar rekomendasi prediksi. Model ini dapat diterapkan dalam sistem e-health, aplikasi mobile, maupun sistem informasi rumah sakit sebagai alat bantu skrining awal.

Selain itu, insight yang dihasilkan dari model dapat digunakan oleh pembuat kebijakan untuk merancang program pencegahan yang lebih tepat sasaran.

---
## Referensi
[1] International Diabetes Federation. “IDF Diabetes Atlas, 10th ed.,” 2021.

[Online]. Available: https://www.diabetesatlas.org

[2] Kementerian Kesehatan RI. “Hasil Utama Riskesdas 2018,” Badan Penelitian dan Pengembangan Kesehatan, Jakarta, 2019.

[3] N. Kumar et al., “Diabetes Detection using Machine Learning: A Review,” Materials Today: Proceedings, vol. 61, pp. 436–441, 2022. [Online]. Available: https://doi.org/10.1016/j.matpr.2022.05.087


# Business Understanding
---
**Problem Statements**
1. Banyak individu yang tidak menyadari bahwa mereka memiliki risiko tinggi terkena diabetes hingga munculnya gejala serius atau komplikasi. Hal ini menyebabkan keterlambatan dalam diagnosis dan pengobatan.
2. Deteksi dini terhadap risiko diabetes sering kali membutuhkan pemeriksaan medis lanjutan yang memakan waktu dan biaya, yang tidak semua orang dapat akses.
3. Dalam dunia medis, dibutuhkan metode alternatif yang cepat, akurat, dan efisien dalam memprediksi risiko diabetes berdasarkan data yang mudah dikumpulkan.

**Goals**
1. Mengembangkan model prediksi berbasis machine learning (Logistic Regression) yang mampu mengidentifikasi individu dengan risiko tinggi terkena diabetes secara dini menggunakan data gejala dan demografi.
2. Menyediakan alat bantu diagnosis yang lebih murah dan cepat sebagai alternatif awal sebelum melakukan tes laboratorium lanjutan.
3. Menyediakan sistem berbasis data yang dapat diintegrasikan dalam sistem informasi rumah sakit atau aplikasi kesehatan untuk membantu pengambilan keputusan medis.

**Solution Statements**
1. Menerapkan algoritma Logistic Regression sebagai baseline model karena kesederhanaannya dan interpretabilitasnya dalam masalah klasifikasi biner seperti deteksi diabetes.
2. Melakukan evaluasi performa model menggunakan metrik seperti accuracy, precision, recall, dan F1-score untuk memastikan bahwa model memiliki kinerja yang dapat diandalkan dalam mendeteksi risiko diabetes.
3. (Improvement)
Model Logistic Regression dapat disempurnakan melalui:
- Hyperparameter tuning, seperti pengaturan C (regularization strength) dan penalty untuk menghindari overfitting atau underfitting.
- Feature selection untuk memastikan hanya fitur relevan yang digunakan dalam pelatihan model.

--- 
# Data Undestanding

Dataset yang digunakan dalam proyek ini adalah [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset) yang diperoleh dari situs [Kaggle](https://www.kaggle.com/). Dataset ini berisi data medis dari pasien, termasuk parameter-parameter laboratorium yang umum digunakan dalam diagnosis dan pemantauan diabetes. Tujuan penggunaan dataset ini adalah untuk membangun model klasifikasi yang dapat memprediksi apakah seorang pasien mengidap diabetes atau tidak.

Dataset ini terdiri dari 14 kolom, yaitu 13 fitur input dan 1 kolom target (CLASS). Data ini mencakup informasi penting seperti usia, jenis kelamin, kadar urea, kreatinin, HbA1c, dan profil lipid. Total terdapat 1000 baris data.

Tautan sumber data: https://www.kaggle.com/datasets/marshalpatel3558/diabetes-prediction-dataset-legit-dataset

---
### **Penjelasan Variabel pada Dataset:**
- `ID`: Nomor identitas data
- `No_Pation`: Nomor pasien
- `Gender`: Jenis kelamin pasien
- `AGE`: Usia pasien
- `Urea`: Kadar urea dalam darah
- `Cr`: Kadar kreatinin
- `HbA1c`: Kadar HbA1c (indikator rata-rata gula darah)
- `Chol`: Total kolesterol
- `TG`: Trigliserida
- `HDL`: High-Density Lipoprotein (kolesterol baik)
- `LDL`: Low-Density Lipoprotein (kolesterol jahat)
- `VLDL`: Very-Low-Density Lipoprotein
- `BMI`: Indeks massa tubuh
- `CLASS`: Label target (positif atau negatif diabetes)

**Dataset ini berisi 1000 baris data.**
**Dan ada 14 kolom yang terdiri dari:**
- Numerik (12 kolom):
    - 8 kolom bertipe float64: Urea, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI
    - 4 kolom bertipe int64: ID, No_Pation, AGE, Cr
- Kategorikal (2 kolom):
    - Gender: kemungkinan berisi 'Male' / 'Female'
    - CLASS: target label seperti 'Diabetic', 'Prediabetic' dan 'Non-Diabetic'

---
## 1. EDA - Bivariate
Dengan mengamati hubungan fitur terhadap kolom CLASS, didapatkan insight sebagai berikut:

- Mayoritas pasien berada pada kelas Y (penderita diabetes), menunjukkan ketidakseimbangan kelas.
- Rata-rata usia pasien kelas Y lebih tinggi dibandingkan kelas lainnya, berkisar di usia 55 tahun.
- BMI pasien kelas Y cenderung lebih tinggi dan terkonsentrasi pada rentang 25–35, mengindikasikan risiko obesitas yang berkaitan dengan diabetes.

---
## 2. EDA - Multivariate
Setelah melihat hubungan antar fitur numerik dan terhadap kelas CLASS, diperoleh insight berikut:

Urea dan Cr memiliki korelasi tertinggi sebesar 0.62.
HbA1c memiliki korelasi positif sedang terhadap BMI, sekitar 0.41.
Pasien dengan kelas Y umumnya memiliki kombinasi nilai AGE tinggi, BMI tinggi, dan HbA1c tinggi.
Distribusi kelas Y mendominasi pada hampir semua kombinasi fitur numerik utama.

---
## 3. Check Missing Values, Duplicated dan Outliers

Tidak terdapat nilai kosong dan duplikat satupun pada data.
Namun terdapat beberapa outlier yang terlihat disini seperti:

- AGE: 98 outlier(s)
- Urea: 65 outlier(s)
- Cr: 52 outlier(s)
- HbA1c: 6 outlier(s)
- Chol: 27 outlier(s)
- TG: 55 outlier(s)
- HDL: 50 outlier(s)
- LDL: 11 outlier(s)
- VLDL: 74 outlier(s)
- BMI: 3 outlier(s)

dan ini harus segera diatasi.

---
# Data Preparation

Pada tahap ini, dilakukan 6 tahapan data preparation secara berurutan agar data siap digunakan untuk proses modeling dan hasil prediksi yang optimal. Tahapan tersebut meliputi:

1. Mengatasi Outlier
Pertama-tama, dilakukan identifikasi dan penanganan outlier pada data. Outlier adalah nilai ekstrem yang dapat mengganggu distribusi data dan menurunkan performa model. Dengan mengatasi outlier, model dapat belajar dari pola data yang lebih representatif.

2. Encoding Data
Selanjutnya, dilakukan encoding pada fitur yang bersifat kategorikal agar dapat diolah oleh algoritma machine learning yang biasanya hanya menerima input numerik. Proses ini mengubah data kategori menjadi format numerik yang dapat dipahami oleh model.

3. Menghapus Nilai NaN
Untuk memastikan bahwa data target CLASS tidak mengandung nilai kosong, dilakukan penghapusan semua baris yang memiliki nilai NaN pada kolom ini menggunakan fungsi dropna(). Langkah ini penting agar model prediksi tidak menerima data dengan label yang tidak diketahui. Setelah proses ini, jumlah data tersisa adalah 676 baris dengan 14 kolom.

4. Splitting Data
Data kemudian dibagi menjadi data training dan data testing dengan proporsi tertentu. Tujuan pemisahan ini adalah untuk melatih model pada data training dan menguji performanya secara objektif pada data testing yang belum pernah dilihat model sebelumnya.

5. Standarisasi
Setelah itu, dilakukan standarisasi pada fitur numerik untuk mengubah data ke skala yang sama dengan mean nol dan standar deviasi satu. Proses ini penting agar fitur dengan rentang nilai berbeda tidak mendominasi proses training, terutama untuk algoritma sensitif terhadap skala data.

6. SMOTE (Synthetic Minority Over-sampling Technique)
Tahap terakhir adalah penyeimbangan kelas menggunakan teknik SMOTE. Karena dataset ini mengalami ketidakseimbangan kelas target, SMOTE digunakan untuk membuat sampel sintetis pada kelas minoritas sehingga model tidak bias terhadap kelas mayoritas dan performa prediksi pada kelas minoritas meningkat.

---
### **Alasan Tahapan Data Preparation diperlukan**
- Mengatasi Outlier: Mencegah nilai ekstrem mengganggu proses pelatihan dan memperbaiki representasi data.
- Encoding Data: Mengubah data kategorikal menjadi format numerik agar dapat diproses model.
- Menghapus Nilai NaN: Menghindari error dan bias saat pelatihan model, karena model tidak dapat belajar dari data tanpa label (CLASS) yang lengkap.
- Splitting Data: Menjaga objektivitas evaluasi model dengan menggunakan data yang belum pernah dilihat selama pelatihan.
- Standarisasi: Menyamakan skala fitur agar setiap fitur berkontribusi seimbang terhadap model.
- SMOTE: Mengatasi ketidakseimbangan kelas untuk meningkatkan kemampuan model dalam mengenali kelas minoritas.

---
## 1. Mengatasi Outliers

Strategi penanganan Outliers menggunakan metode IQR (Interquartile Range) untuk deteksi. Dataset sebelum dilakukan pengahapusan outliers dengan IQR adalah sebanyak 1000 data dan sekarang Outlier sudah diatasi total data saat kini sebanyak 678.

## 2. Encoding Data

**Encoding kolom Gender**
Tahap ini adalah proses encoding (pengkodean) data kategorikal menjadi data numerik agar bisa digunakan dalam model machine learning. Penjelasannya sebagai berikut:

- Di sini, nilai 'F' (Female/perempuan) diubah menjadi 0
- Nilai 'M' (Male/laki-laki) diubah menjadi 1

Tujuan: Model machine learning tidak bisa memproses string seperti 'F' atau 'M'. Maka, kita ubah ke angka agar bisa diproses secara matematis.

**Encoding kolom CLASS**

- 'N' (kemungkinan berarti "Normal" atau "Tidak Positif") menjadi -1
- 'P' (kemungkinan berarti "Pre-diabetes" atau kondisi antara normal dan diabetes) menjadi 0
- 'Y' (kemungkinan berarti "Yes"/positif diabetes) menjadi 1

Tujuan: Sama seperti sebelumnya, kita ingin ubah data kategorikal ke numerik agar bisa digunakan dalam model prediksi atau klasifikasi.

## 3. Menghapus Nilai NaN

Dalam proses pembersihan data (data cleaning), salah satu langkah penting adalah menghapus data yang tidak lengkap atau memiliki nilai yang hilang (missing values). Khususnya pada tahap ini, dilakukan penghapusan baris yang memiliki nilai NaN pada kolom CLASS, yang merupakan target variabel (label) dalam analisis atau pemodelan prediksi.

## 4. Splitting Data

Splitting data adalah proses membagi data menjadi dua bagian, yaitu data latih (training) dan data uji (testing), dengan tujuan agar model machine learning bisa belajar dari satu bagian (training) dan diuji kemampuannya pada bagian lain (testing) yang belum pernah dilihat sebelumnya. 

Ini penting agar kita bisa mengevaluasi seberapa baik model memprediksi data baru, serta mencegah model hanya menghafal data tanpa benar-benar memahami polanya.

## 5. Standarisasi

Tujuan dari standarisasi adalah proses mengubah nilai-nilai data agar semua fitur memiliki skala yang sama, biasanya dengan membuat rata-ratanya menjadi 0 dan standar deviasinya 1. Ini penting karena beberapa algoritma machine learning sangat sensitif terhadap perbedaan skala antar fitur, sehingga standarisasi membantu model belajar lebih adil dan akurat dari semua variabel yang ada.

## 6. SMOTE

SMOTE adalah teknik yang digunakan untuk menyeimbangkan jumlah data antar kelas dalam masalah klasifikasi, terutama saat satu kelas memiliki jauh lebih sedikit data dibanding yang lain. Caranya dengan membuat contoh data baru (sintetis) dari kelas yang jumlahnya sedikit, sehingga model machine learning tidak berat sebelah dan bisa mengenali pola dari semua kelas dengan lebih adil dan akurat.

---
# Modeling

**Regresi logistik** adalah algoritma pembelajaran mesin terbimbing yang digunakan untuk tugas klasifikasi yang bertujuan untuk memprediksi probabilitas bahwa suatu contoh termasuk dalam kelas tertentu atau tidak. Regresi logistik adalah algoritma statistik yang menganalisis hubungan antara dua faktor data.

Model ini digunakan dengan parameter default, yang berarti tidak dilakukan perubahan pada nilai parameter saat inisialisasi model. Parameter default yang digunakan pada LogisticRegression() dari library scikit-learn, di antaranya:

- `penalty='l2'` : Regularisasi L2 (Ridge).
- `C=1.0` : Invers dari kekuatan regularisasi.
- `solver='lbfgs'` : Algoritma optimisasi.
- `max_iter=100` : Jumlah maksimum iterasi untuk konvergensi.

**Cara kerjanya:**

dimulai dengan menghitung gabungan dari semua fitur input dan bobotnya. Hasil dari perhitungan ini kemudian diubah menjadi nilai probabilitas menggunakan fungsi khusus yang disebut fungsi sigmoid.

Nilai probabilitas tersebut menunjukkan seberapa besar kemungkinan suatu data termasuk ke dalam kelas tertentu. Jika nilainya lebih tinggi dari ambang batas (biasanya 0.5), maka data diklasifikasikan ke dalam satu kelas (misalnya, 1). Jika lebih rendah, maka diklasifikasikan ke kelas lainnya (misalnya, 0).

Saat proses pelatihan, model belajar untuk menyesuaikan bobot-bobotnya agar bisa memprediksi label dengan lebih akurat. Logistic Regression juga menggunakan teknik regularisasi untuk mencegah model terlalu mengikuti pola data latih, sehingga tetap bekerja baik pada data baru.

Secara keseluruhan, Logistic Regression bekerja dengan cepat, sederhana, dan sangat cocok untuk masalah klasifikasi biner yang memiliki hubungan yang cukup linier antara fitur dan target.

**Kelebihan** dan **Kekurangan** Logistic Regression

- Kelebihan:
    - Model sederhana dan mudah diinterpretasikan.
    - Cepat dalam proses training dan prediksi.
    - Cocok untuk masalah klasifikasi biner dengan hubungan linear antara fitur dan target.

- Kekurangan:
    - Kurang efektif jika data memiliki pola non-linear.
    - Sensitif terhadap multikolinearitas dan fitur yang tidak distandarisasi.

---
# Evaluation

Dalam proyek ini, dilakukan evaluasi terhadap model klasifikasi menggunakan beberapa metrik utama, yaitu:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Metrik-metrik ini dipilih karena sangat umum digunakan dalam kasus klasifikasi dan mampu memberikan gambaran menyeluruh mengenai performa model, baik secara keseluruhan maupun per kelas.

Confusion Matrix:
[[  9   3   1]
 [  0  10   4]
 [  2   3 104]]

Ini menunjukkan bahwa:

- Kelas -1: 9 benar, 3 salah ke kelas 0, 1 salah ke kelas 1
- Kelas 0: 10 benar, 4 salah ke kelas 1
- Kelas 1: 104 benar, 5 salah ke kelas -1 atau 0

Classification Report:
              precision    recall  f1-score   support

          -1       0.82      0.69      0.75        13
           0       0.62      0.71      0.67        14
           1       0.95      0.95      0.95       109

    accuracy                           0.90       136
   macro avg       0.80      0.79      0.79       136
weighted avg       0.91      0.90      0.91       136

## **Interpretasi:**

- Akurasi tinggi (90%) menunjukkan model secara umum dapat memprediksi dengan baik.
- Kelas 1 memiliki kinerja terbaik karena memiliki jumlah data terbanyak (109), dengan precision, recall, dan F1-score semua 0.95.
- Kelas 0 memiliki precision yang paling rendah (0.62), artinya model sering salah mengklasifikasikan data lain sebagai kelas 0.
- F1-score macro average (0.79) lebih rendah dari weighted average (0.91), menunjukkan model cenderung lebih baik pada kelas mayoritas (kelas 1) dibandingkan kelas minoritas (-1 dan 0).

## **Kesimpulan**
Penelitian ini bertujuan untuk menjawab tantangan dalam deteksi dini risiko diabetes, terutama bagi individu yang tidak memiliki akses terhadap pemeriksaan medis lanjutan dan sering kali tidak menyadari kondisi kesehatannya hingga muncul gejala serius. Dengan memanfaatkan data gejala dan demografi yang relatif mudah dikumpulkan, model Logistic Regression berhasil dikembangkan sebagai solusi prediktif awal yang cepat, murah, dan dapat diakses luas.

Hasil evaluasi menunjukkan bahwa model memiliki akurasi keseluruhan sebesar 90% dan F1-score yang sangat tinggi (0.95) pada kelas mayoritas (label 1). Namun, performa pada kelas minoritas (label -1 dan 0) masih perlu ditingkatkan, dengan F1-score masing-masing sebesar 0.75 dan 0.67. Hal ini mencerminkan tantangan klasik dalam klasifikasi data tidak seimbang, yang perlu menjadi perhatian utama dalam pengembangan lebih lanjut.

Model ini menjawab secara langsung problem statement yang telah diuraikan, dengan memberikan alternatif sistem prediksi awal yang dapat digunakan sebagai alat bantu diagnosis sebelum melakukan pemeriksaan laboratorium yang lebih kompleks. Selain itu, pendekatan ini sangat potensial untuk diintegrasikan dalam sistem informasi rumah sakit atau aplikasi kesehatan berbasis data, guna mendukung pengambilan keputusan medis dan meningkatkan kesadaran masyarakat akan risiko diabetes.

Untuk meningkatkan performa model secara lebih seimbang ke semua kelas, disarankan untuk melakukan:
- Penyeimbangan data, seperti oversampling pada kelas minoritas atau undersampling kelas mayoritas.
- Pengaturan class weight dalam algoritma Logistic Regression agar model lebih sensitif terhadap kelas-kelas dengan jumlah data lebih sedikit.

Dengan pengembangan lebih lanjut, model ini dapat menjadi fondasi bagi sistem deteksi dini diabetes yang lebih akurat dan inklusif.
