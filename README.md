# Proyek Machine Learning - Deteksi Diabetes Menggunakan Pima Indians Dataset

## Domain Proyek

Penyakit diabetes melitus adalah salah satu penyakit tidak menular yang menjadi perhatian utama kesehatan global. Menurut World Health Organization (WHO), sekitar 422 juta orang di seluruh dunia menderita diabetes, dan jumlah ini terus meningkat tiap tahunnya. Pendeteksian dini terhadap potensi diabetes sangat penting untuk mencegah komplikasi yang lebih serius di masa depan. Dengan meningkatnya jumlah data kesehatan dan kemajuan teknologi, machine learning menjadi pendekatan yang sangat potensial untuk membangun sistem deteksi dini diabetes berbasis data.

Dataset yang digunakan dalam proyek ini adalah **Pima Indians Diabetes Dataset** dari Kaggle yang berasal dari National Institute of Diabetes and Digestive and Kidney Diseases. Dataset ini terdiri dari data medis perempuan keturunan Pima Indian berusia di atas 21 tahun.

**Referensi:**

- World Health Organization, "Diabetes", 2023. \[Online]. Available: [https://www.who.int/news-room/fact-sheets/detail/diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)
- Kaggle, Pima Indians Diabetes Database. \[Online]. Available: [https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

## Business Understanding

### Problem Statements

- Bagaimana memprediksi apakah seseorang terdiagnosis diabetes berdasarkan data medis yang tersedia?
- Fitur medis apa saja yang paling berpengaruh terhadap diagnosis diabetes?

### Goals

- Mengembangkan model klasifikasi untuk mendeteksi kemungkinan seseorang terdiagnosis diabetes.
- Mengukur dan membandingkan performa model-model machine learning terhadap data tersebut.

### Solution statements

- Menerapkan dua algoritma klasifikasi: **Random Forest** dan **Voting Classifier** berbasis ensemble.
- Melakukan **hyperparameter tuning** pada Random Forest menggunakan `RandomizedSearchCV`.
- Melakukan evaluasi model menggunakan metrik **accuracy**, **precision**, **recall**, dan **f1-score**.

## Data Understanding

Dataset dapat diunduh di sini: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

Dataset terdiri dari 768 data pasien dengan 9 fitur:

### Variabel-variabel:

- `Pregnancies`: Jumlah kehamilan
- `Glucose`: Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral
- `BloodPressure`: Tekanan darah diastolik (mm Hg)
- `SkinThickness`: Ketebalan lipatan kulit trisep (mm)
- `Insulin`: Kadar insulin serum 2 jam (mu U/ml)
- `BMI`: Indeks massa tubuh (kg/m^2)
- `DiabetesPedigreeFunction`: Fungsi silsilah diabetes
- `Age`: Usia (tahun)
- `Outcome`: Diagnosis (0 = tidak diabetes, 1 = diabetes)

### EDA dan Temuan:

- Terdapat nilai 0 pada fitur `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, dan `BMI` yang tidak realistis dan perlu ditangani.
- Terdapat ketidakseimbangan kelas (positif: 268, negatif: 500).
- Korelasi paling kuat terhadap outcome ditemukan pada `Glucose`, `BMI`, dan `Age`.

## Data Preparation

### Teknik yang digunakan:

- Mengganti nilai 0 pada fitur medis dengan nilai median dari fitur tersebut.
- Melakukan normalisasi fitur menggunakan `StandardScaler`.
- Membagi data menjadi data latih (80%) dan data uji (20%) dengan `train_test_split`.
- Mengatasi ketidakseimbangan data menggunakan `SMOTE`
- Penanganan ketidakseimbangan kelas menggunakan SMOTE: Diterapkan SMOTE (Synthetic Minority Oversampling Technique) pada data pelatihan untuk memperbanyak sampel pada kelas minoritas (positif diabetes) dengan data sintetis.
- Seleksi fitur menggunakan RFE (Recursive Feature Elimination): Digunakan dengan estimator `LogisticRegression` untuk memilih 5 fitur terbaik berdasarkan kontribusinya terhadap target.

### Alasan:

- Nilai 0 bukan representasi valid untuk fitur medis, sehingga perlu diganti dengan median agar distribusi tetap terjaga.
- Normalisasi dibutuhkan karena beberapa model sensitif terhadap skala data.
- Pembagian data penting untuk menghindari overfitting dan mengevaluasi generalisasi model.
- Seleksi fitur (RFE) membantu menyederhanakan model, mengurangi risiko overfitting, dan meningkatkan interpretabilitas.
- SMOTE lebih efektif daripada `class_weight='balanced'` dalam kondisi ketidakseimbangan kelas yang tinggi karena menghasilkan distribusi yang lebih seimbang secara eksplisit pada data pelatihan.

## Modeling

### Penjelasan Konseptual Algoritma

##### Random Forest

Random Forest adalah algoritma ensemble berbasis decision tree yang membangun banyak pohon keputusan (trees) selama pelatihan dan menghasilkan prediksi berdasarkan mayoritas voting (untuk klasifikasi) atau rata-rata (untuk regresi). Setiap pohon dilatih dengan subset acak dari data dan fitur (bagging), yang membuat model lebih tahan terhadap overfitting dibandingkan dengan satu decision tree. Randomness dalam pemilihan fitur dan data meningkatkan generalisasi model.

##### Logistic Regression

Logistic Regression adalah algoritma klasifikasi linier yang memodelkan probabilitas suatu sampel termasuk dalam kelas tertentu menggunakan fungsi sigmoid. Model ini mencari garis pemisah linier terbaik di ruang fitur untuk membedakan antara dua kelas, dan sangat efektif jika hubungan antara fitur dan target bersifat linier.

##### Support Vector Classifier (SVC)

SVC bekerja dengan mencari hyperplane terbaik yang memisahkan dua kelas data dengan margin maksimum. SVC menggunakan kernel (seperti linear atau RBF) untuk memetakan data ke dimensi yang lebih tinggi jika data tidak dapat dipisahkan secara linier. Dalam proyek ini digunakan probability=True agar dapat digunakan dalam Voting Classifier berbasis soft voting.

##### Voting Classifier

Voting Classifier adalah algoritma ensemble yang menggabungkan prediksi dari beberapa model klasifikasi dasar. Dalam proyek ini, digunakan tiga model: Random Forest, Logistic Regression, dan Support Vector Classifier (SVC). Model ini menggunakan pendekatan soft voting, di mana probabilitas prediksi dari masing-masing model digabungkan (dirata-rata), dan kelas dengan probabilitas tertinggi dipilih sebagai prediksi akhir. Dengan menggabungkan model dengan karakteristik berbeda, Voting Classifier dapat meningkatkan performa dan mengurangi bias model tunggal.

### Model yang digunakan:

1. **Random Forest**: Model baseline dengan tuning parameter seperti `n_estimators`, `max_depth`, dan `min_samples_split` menggunakan `RandomizedSearchCV`.
2. **Voting Classifier**: Menggabungkan Random Forest, Logistic Regression, dan SVC untuk meningkatkan performa dan stabilitas model.

### Model Parameter Eksplisit:

- Random forest

```python
rf = RandomForestClassifier(
    n_estimators=180,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=True,
    random_state=42
)
```

- SVM

```python
svm = SVC(
    probability=True,
    random_state=42
)
```

- Logistic Regression

```python
lr = LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

- KNN

```python
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski'
)

```

### Parameter Eksplisit Voting Classifier

```python
log_reg = LogisticRegression(random_state=42, max_iter=1000)
svm_clf = SVC(probability=True, random_state=42)
rf_clf = best_rf  # model Random Forest hasil tuning

voting_clf = VotingClassifier(
    estimators=[('lr', log_reg), ('svm', svm_clf), ('rf', rf_clf)],
    voting='soft'  # voting berdasarkan probabilitas
)
```

### Hyperparameter Tuning:

Dilakukan pencarian acak dengan `RandomizedSearchCV` selama 50 iterasi terhadap Random Forest untuk mendapatkan parameter terbaik.

### Parameter terbaik:

```python
{
  'bootstrap': True,
  'max_depth': None,
  'min_samples_leaf': 1,
  'min_samples_split': 2,
  'n_estimators': 180
}
```

Best Cross-Validation F1-Score: `0.8429`

## Evaluation

### Metrik yang digunakan:

- **Accuracy**: proporsi prediksi benar terhadap total data
- **Precision**: proporsi prediksi positif yang benar (TP / (TP + FP))
- **Recall (Sensitivity)**: proporsi positif yang terdeteksi benar (TP / (TP + FN))
- **F1-score**: harmonisasi antara precision dan recall (2 \* (precision \* recall) / (precision + recall))

### Random Forest (Tanpa Tuning)

**Confusion Matrix:**

```
[[76 24]
 [16 38]]
```

**Classification Report:**

```
              precision    recall  f1-score   support

           0       0.83      0.76      0.79       100
           1       0.61      0.70      0.66        54

    accuracy                           0.74       154
   macro avg       0.72      0.73      0.72       154
weighted avg       0.75      0.74      0.74       154
```

### Random Forest dengan Threshold Tuning (0.24)

**Confusion Matrix:**

```
[[61 39]
 [ 4 50]]
```

**Classification Report:**

```
              precision    recall  f1-score   support

           0       0.94      0.61      0.74       100
           1       0.56      0.93      0.70        54

    accuracy                           0.72       154
   macro avg       0.75      0.77      0.72       154
weighted avg       0.81      0.72      0.73       154
```

### Logistic Regression

**Confusion Matrix:**

```
[[74 26]
 [17 37]]
```

**Classification Report:**

```
              precision    recall  f1-score   support

           0       0.81      0.74      0.77       100
           1       0.59      0.69      0.63        54

    accuracy                           0.72       154
   macro avg       0.70      0.71      0.70       154
weighted avg       0.73      0.72      0.72       154
```

### SVM

**Confusion Matrix:**

```
[[75 25]
 [17 37]]
```

**Classification Report:**

```
              precision    recall  f1-score   support

           0       0.82      0.75      0.78       100
           1       0.60      0.69      0.64        54

    accuracy                           0.73       154
   macro avg       0.71      0.72      0.71       154
weighted avg       0.74      0.73      0.73       154
```

### KNN

**Confusion Matrix:**

```
[[68 32]
 [15 39]]
```

**Classification Report:**

```
              precision    recall  f1-score   support

           0       0.82      0.68      0.74       100
           1       0.55      0.72      0.62        54

    accuracy                           0.69       154
   macro avg       0.68      0.70      0.68       154
weighted avg       0.72      0.69      0.70       154
```

### Voting Classifier

**Confusion Matrix:**

```
[[77 23]
 [14 40]]
```

**Classification Report:**

```
              precision    recall  f1-score   support

           0       0.85      0.77      0.81       100
           1       0.63      0.74      0.68        54

    accuracy                           0.76       154
   macro avg       0.74      0.76      0.75       154
weighted avg       0.77      0.76      0.76       154
```

### Kesimpulan:

| Model                   | Accuracy | Precision (1) | Recall (1) | F1-Score (1) |
| ----------------------- | -------- | ------------- | ---------- | ------------ |
| Support Vector Machine  | 0.72     | 0.60          | 0.69       | 0.64         |
| K-Nearest Neighbors     | 0.69     | 0.55          | 0.72       | 0.62         |
| Logistic Regression     | 0.72     | 0.59          | 0.69       | 0.63         |
| Random Forest (default) | 0.74     | 0.61          | 0.70       | 0.66         |
| RF + Threshold 0.24     | 0.72     | 0.56          | **0.93**   | 0.70         |
| Voting Classifier       | **0.76** | 0.63          | 0.74       | 0.68         |

Model yang dikembangkan berhasil mencapai performa yang baik dalam mendeteksi diabetes pada data pasien. Dengan teknik tuning dan ensemble, performa model meningkat secara signifikan terutama dari segi recall dan f1-score, yang sangat penting dalam konteks diagnosis medis untuk menghindari kesalahan negatif (false negative).

### Hubungan Evaluasi Model dengan Business Understanding

Model yang dikembangkan bertujuan untuk mendeteksi kemungkinan seseorang menderita diabetes berdasarkan data medis mereka. Berdasarkan hasil evaluasi, model Voting Classifier menunjukkan performa terbaik secara keseluruhan dengan accuracy 76%, precision 63%, dan recall 74% pada kelas positif (pasien dengan diabetes). Ini menunjukkan bahwa model mampu mengidentifikasi sebagian besar pasien yang benar-benar memiliki diabetes, yang sangat penting dalam konteks medis untuk mencegah kasus yang tidak terdeteksi (false negatives). Model Random Forest dengan threshold tuning memberikan recall tertinggi (93%), yang berarti sangat sensitif dalam menangkap kasus diabetes.

Dampak dari model ini terhadap business understanding adalah sebagai berikut:

- Menjawab Problem Statement 1: Model berhasil memberikan prediksi diagnosis diabetes berdasarkan fitur medis, terbukti dengan performa evaluasi yang memuaskan.

- Menjawab Problem Statement 2: Dengan fitur-fitur seperti Glucose, BMI, dan Age yang dipilih melalui proses RFE dan analisis korelasi, kita juga memperoleh wawasan mengenai faktor medis yang berkontribusi terhadap diagnosis.

- Mencapai Goals: Model yang dikembangkan menunjukkan hasil evaluasi yang kompetitif dan bermanfaat untuk tujuan deteksi dini, sekaligus menunjukkan peningkatan performa melalui teknik tuning dan ensemble.
