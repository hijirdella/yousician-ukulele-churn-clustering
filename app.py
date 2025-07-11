import streamlit as st
import pandas as pd
import joblib

# === Konfigurasi Halaman ===
st.set_page_config(page_title="Ukulele by Yousician - Churned User Clustering", layout="wide")
st.title("🎵 Ukulele by Yousician - Churned User Clustering App")

# === Mapping Label Klaster ===
label_mapping = {
    0: "Focused Achievers",
    1: "Advanced Explorers",
    2: "Casual Dropouts"
}

# === Pilih Mode Input ===
input_mode = st.radio("Pilih Mode Input:", ["👤 Input Manual (1 Pengguna)", "📁 Upload CSV (Batch Pengguna)"])

# === MODE 1: INPUT MANUAL UNTUK 1 USER ===
if input_mode == "👤 Input Manual (1 Pengguna)":
    st.subheader("Masukkan Data Pengguna")

    # Load model dan scaler
    model_1user = joblib.load("kmeans_churned_model_1user.pkl")
    scaler_1user = joblib.load("scaler_clustering_1user.pkl")
    feature_names_1user = joblib.load("clustering_feature_names_1user.pkl")

    # Input manual
    user_input = {}
    for feat in feature_names_1user:
        user_input[feat] = st.number_input(f"{feat}", value=0.0)

    if st.button("🧭 Prediksi Cluster"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler_1user.transform(input_df)
        cluster = model_1user.predict(input_scaled)[0]
        label = label_mapping.get(cluster, f"Cluster {cluster}")
        st.success(f"✅ User ini diprediksi berada pada Cluster: **{label}**")

# === MODE 2: UPLOAD CSV UNTUK BANYAK USER ===
else:
    st.subheader("Unggah File CSV")

    # Load model dan scaler
    model_batch = joblib.load("kmeans_churned_batch_model.pkl")
    scaler_batch = joblib.load("scaler_clustering_batch.pkl")
    feature_names_batch = joblib.load("clustering_feature_names_batch.pkl")

    uploaded_file = st.file_uploader("Unggah file CSV dengan fitur yang sesuai", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📄 Preview Data:", df.head())

            # Validasi fitur
            if not all(feat in df.columns for feat in feature_names_batch):
                st.error("❌ Kolom fitur tidak lengkap. Harap pastikan nama kolom sesuai.")
            else:
                X = df[feature_names_batch]
                X_scaled = scaler_batch.transform(X)
                clusters = model_batch.predict(X_scaled)
                df['Predicted_Cluster'] = clusters
                df['Cluster_Label'] = df['Predicted_Cluster'].map(label_mapping)

                st.success("📈 Cluster berhasil diprediksi untuk semua pengguna.")
                st.dataframe(df)

                # Tombol download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Hasil", csv, "clustered_users.csv", "text/csv")
        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan saat membaca file: {e}")
