import streamlit as st
import cv2
import os
import time

# Tentukan folder untuk menyimpan dataset
dataset_folder = "dataset/test/images"
os.makedirs(dataset_folder, exist_ok=True)

# Inisialisasi status tombol di session_state
if "capture" not in st.session_state:
    st.session_state.capture = False
if "stop" not in st.session_state:
    st.session_state.stop = False

# Fungsi untuk menangkap gambar secara real-time
def capture_images():
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        st.error("Tidak dapat mengakses webcam. Periksa koneksi kamera.")
        return

    stframe = st.empty()  # Tempat untuk menampilkan video stream

    col1, col2 = st.columns(2)  # Buat layout 2 kolom untuk tombol
    with col1:
        if st.button("Capture Image", key="capture_btn"):
            st.session_state.capture = True  # Set status capture ke True
    with col2:
        if st.button("Stop Capture", key="stop_btn"):
            st.session_state.stop = True  # Set status stop ke True

    while not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membaca gambar dari webcam")
            break

        # Menampilkan video stream
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # Jika tombol Capture ditekan
        if st.session_state.capture:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            img_path = os.path.join(dataset_folder, f"image_{timestamp}.jpg")
            cv2.imwrite(img_path, frame)
            st.success(f"Image saved as {img_path}")
            st.session_state.capture = False  # Reset status capture

        time.sleep(0.1)  # Delay kecil agar tidak terlalu cepat

    cap.release()
    stframe.empty()

# Tampilan antarmuka Streamlit
st.title("Real-Time Capture and Save Image")
capture_images()
