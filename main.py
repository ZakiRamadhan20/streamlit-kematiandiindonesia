import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Mengatur konfigurasi halaman
st.set_page_config(page_title="Analisis Penyebab Kematian", layout="wide")

# Judul aplikasi
st.title("Prediksi dan Pengelompokan Faktor Risiko Penyakit Jantung")
st.subheader("Evaluasi Performa Random Forest dan K-Means")

# Membuat dua kolom dengan pembagian 30:70
col1, col2 = st.columns([3, 7])  # Kolom kiri 30%, kolom kanan 70%

# Bagian untuk mengunggah dataset di kolom kiri
with col1:
    st.markdown("""
    <div style="background-color: #001f3f; color: white; padding: 20px; border-radius: 5px;">
        <h3 style="text-align: center;">Upload Dataset</h3>
        <p style="text-align: center;">Drag and drop file here or browse</p>
    </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])

    # Pesan untuk mengunggah file
    st.info("Silakan unggah file dataset untuk melanjutkan.")

# Bagian untuk menampilkan data di kolom kanan
with col2:
    if uploaded_file is not None:
        # Membaca dataset
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)

        # Data Overview
        st.subheader("Data Overview")
        st.write("First 10 Rows of Data")
        st.dataframe(data.head(10))

        # Data Description
        st.subheader("Data Description")
        st.write(data.describe())

        # Data Info
        st.subheader("Data Info")
        st.write(data.info())

        # Memeriksa distribusi kelas
        target_column = st.selectbox("Pilih kolom target untuk klasifikasi:", options=data.columns)
        if target_column in data.columns:
            st.subheader("Distribusi Kelas")
            st.write(data[target_column].value_counts())
            st.bar_chart(data[target_column].value_counts())  # Menampilkan grafik distribusi kelas

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_data = data.select_dtypes(include=['float64', 'int64'])

        if not numeric_data.empty:
            plt.figure(figsize=(12, 10))  # Ukuran figure yang lebih besar
            heatmap = sns.heatmap(numeric_data.corr(), annot=True, fmt=".2f", cmap='coolwarm', 
                                   square=True, cbar_kws={"shrink": .8}, linewidths=0.5)
            plt.title('Correlation Heatmap', fontsize=16, color='#003366')  # Judul dengan warna biru navy
            plt.xticks(rotation=45, ha='right')  # Memutar label sumbu x
            plt.yticks(rotation=0)  # Memutar label sumbu y
            st.pyplot(plt)
        else:
            st.warning("Tidak ada kolom numerik untuk menghitung korelasi.")

        # K-Means Clustering
        st.subheader("K-Means Clustering")
        if not numeric_data.empty:
            # Menggunakan metode elbow untuk menentukan jumlah cluster yang optimal
            inertia = []
            k_range = range(1, 11)
            for k in k_range:
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(numeric_data)
                inertia.append(kmeans.inertia_)

            plt.figure(figsize=(10, 6))
            plt.plot(k_range, inertia, marker='o', linestyle='--')
            plt.title('Elbow Method for Optimal k', fontsize=16, color='#003366')  # Judul dengan warna biru navy
            plt.xlabel('Number of Clusters (k)', color='#003366')  # Label sumbu x dengan warna biru navy
            plt.ylabel('Inertia', color='#003366')  # Label sumbu y dengan warna biru navy
            st.pyplot(plt)
        else:
            st.warning("Tidak ada kolom numerik untuk melakukan K-Means Clustering.")

        # Decision Tree Classification
        st.subheader("Decision Tree Classification")
        
        if target_column in data.columns:
            # Pisahkan fitur dan target
            X = data.drop(target_column, axis=1)  # Menghapus kolom target dari fitur
            y = data[target_column]  # Menggunakan kolom target

            # Encoding kolom kategorikal
            X = pd.get_dummies(X, drop_first=True)  # Menggunakan One-Hot Encoding

            # Membagi data menjadi set pelatihan dan pengujian
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Memilih kedalaman maksimum pohon keputusan
            max_depth = st.slider("Pilih Kedalaman Maksimum Decision Tree:", min_value=1, max_value=10, value=3)

            # Membuat model Decision Tree
            model = DecisionTreeClassifier(max_depth=max_depth)  # Menggunakan kedalaman maksimum yang dipilih
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Akurasi Model
            accuracy = accuracy_score(y_test, y_pred) * 100
            st.markdown(f"<h3 style='color: #003366;'>Akurasi Model: {accuracy:.2f}%</h3>", unsafe_allow_html=True)

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            st.write(cm)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)

            # Membuat DataFrame untuk laporan klasifikasi
            report_df = pd.DataFrame(report).transpose()
            report_df = report_df.drop(index=['accuracy', 'macro avg', 'weighted avg'])  # Menghapus rata-rata
            st.table(report_df)  # Menampilkan laporan klasifikasi dalam format tabel

            # Visualisasi Decision Tree
            st.subheader("Visualisasi Decision Tree")
            plt.figure(figsize=(12, 8))
            plot_tree(model, filled=True, feature_names=X.columns, class_names=model.classes_)
            st.pyplot(plt)

        else:
            st.warning(f"Kolom target '{target_column}' tidak ditemukan. Pastikan untuk memilih kolom yang valid.")

