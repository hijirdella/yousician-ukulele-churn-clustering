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

    predicted_churn = st.selectbox("Apakah Predicted_Churn user ini?", [1, 0])

    if st.button("🧭 Prediksi Cluster"):
        if predicted_churn == 1:
            input_df = pd.DataFrame([user_input])
            input_scaled = scaler_1user.transform(input_df)
            cluster = model_1user.predict(input_scaled)[0]
            label = label_mapping.get(cluster, f"Cluster {cluster}")
            st.success(f"✅ User ini diprediksi churn dan berada pada Cluster: **{label}**")
        else:
            st.info("ℹ️ User ini tidak churn. Tidak dilakukan pemetaan klaster.")

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
            st.write("📄 Preview Data (semua user):", df.head())

            # Validasi kolom
            required_columns = set(feature_names_batch + ["Predicted_Churn"])
            if not required_columns.issubset(set(df.columns)):
                st.error(f"❌ File harus memuat kolom: {required_columns}")
            else:
                df_churn = df[df["Predicted_Churn"] == 1].copy()

                if df_churn.empty:
                    st.warning("⚠️ Tidak ada user dengan Predicted_Churn = 1.")
                else:
                    X = df_churn[feature_names_batch]
                    X_scaled = scaler_batch.transform(X)
                    clusters = model_batch.predict(X_scaled)

                    df_churn['Predicted_Cluster'] = clusters
                    df_churn['Cluster_Label'] = df_churn['Predicted_Cluster'].map(label_mapping)

                    st.success(f"📈 {len(df_churn)} user churn berhasil diklasterisasi.")
                    st.dataframe(df_churn)

                    # Tombol download
                    csv = df_churn.to_csv(index=False).encode('utf-8')
                    st.download_button("💾 Download Hasil Churned Users", csv, "churned_clustered_users.csv", "text/csv")

        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan saat membaca file: {e}")
