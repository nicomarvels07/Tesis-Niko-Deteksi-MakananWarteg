import streamlit as st
import cv2
import os
import time

# Tentukan folder untuk menyimpan dataset
dataset_folder = "dataset/test/images"
os.makedirs(dataset_folder, exist_ok=True)

# Inisialisasi status tombol di session_state
if "running" not in st.session_state:
    st.session_state.running = False

# Fungsi untuk menangkap gambar secara real-time tanpa tombol capture
def capture_images():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("Tidak dapat mengakses webcam. Periksa koneksi kamera.")
        return

    stframe = st.empty()  # Tempat untuk menampilkan video stream

    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Real-Time Capture"):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Stop Capture"):
            st.session_state.running = False

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membaca gambar dari webcam")
            break

        # Menampilkan video stream secara real-time
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Simpan gambar secara otomatis setiap 2 detik
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        img_path = os.path.join(dataset_folder, f"image_{timestamp}.jpg")
        cv2.imwrite(img_path, frame)
        st.info(f"📸 Gambar disimpan: {img_path}")
        time.sleep(2)  # Delay pengambilan gambar setiap 2 detik

    cap.release()
    stframe.empty()

# Tampilan antarmuka Streamlit
st.title("📷 Real-Time Image Capture")
capture_images()
