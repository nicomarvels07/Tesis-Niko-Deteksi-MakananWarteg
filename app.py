import streamlit as st
import cv2
import os
import time

# Tentukan folder untuk menyimpan dataset
dataset_folder = "dataset/test/images"
os.makedirs(dataset_folder, exist_ok=True)

# Inisialisasi status running di session_state
if "stop" not in st.session_state:
    st.session_state.stop = False

# Fungsi untuk menangkap gambar secara otomatis saat dijalankan
def auto_capture_images(interval=3, max_duration=30):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        st.error("❌ Tidak dapat mengakses webcam. Periksa koneksi kamera.")
        return

    stframe = st.empty()  # Untuk menampilkan video stream
    st.warning("🚨 **Capturing images automatically...**")
    st.info(f"📸 Gambar akan diambil setiap {interval} detik, selama {max_duration} detik.")

    start_time = time.time()

    while time.time() - start_time < max_duration and not st.session_state.stop:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Gagal membaca gambar dari webcam.")
            break

        # Menampilkan video stream
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Simpan gambar dengan timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        img_path = os.path.join(dataset_folder, f"image_{timestamp}.jpg")
        cv2.imwrite(img_path, frame)
        st.success(f"✅ Gambar tersimpan: {img_path}")
        time.sleep(interval)  # Interval pengambilan gambar

    cap.release()
    stframe.empty()
    st.success("✅ Proses capture selesai.")

# Tampilan antarmuka Streamlit
st.title("🎬 Real-Time Auto Capture Image")

if st.button("⏹ Stop Capture"):
    st.session_state.stop = True
    st.warning("🛑 Capture dihentikan secara manual.")

# Jalankan fungsi capture otomatis
auto_capture_images(interval=3, max_duration=30)
