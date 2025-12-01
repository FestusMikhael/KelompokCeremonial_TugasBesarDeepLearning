import subprocess
import sys
import os
import shutil
import zipfile
import pandas as pd

# INSTALASI OTOMATIS
try:
    import facenet_pytorch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "facenet-pytorch", "--no-deps"])

import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from PIL import Image, ImageOps
import numpy as np

# DOWNLOAD MODEL
def download_model_if_needed():
    url = "https://huggingface.co/spaces/Kenneth16/Tubes_DeepLearning/resolve/main/facenet_best_model.pth"
    filename = "facenet_best_model.pth"
    if not os.path.exists(filename) or os.path.getsize(filename) < 10000:
        try:
            import requests
            response = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except: pass

download_model_if_needed()

# LOAD MODEL
@st.cache_resource
def load_models():
    device = torch.device('cpu')
    try:
        with open('class_names.txt', 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
    except: return None, None, None

    model = InceptionResnetV1(classify=True, pretrained=None, num_classes=len(class_names))
    try:
        state_dict = torch.load('facenet_best_model.pth', map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
    except: return None, None, None
        
    model.to(device)
    model.eval()
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, post_process=False, device=device)
    return model, mtcnn, class_names

model, mtcnn, class_names = load_models()

if model is None:
    st.error("Gagal memuat model. Pastikan file model dan class_names.txt ada.")
    st.stop()

# UI UTAMA
st.set_page_config(page_title="Uji Akurasi FaceNet", page_icon="🎓", layout="wide")
st.title("🎓 Sistem Evaluasi Wajah Mahasiswa")

# Buat Tab Navigasi
tab1, tab2 = st.tabs(["Cek Satu Foto", "Uji Dataset (ZIP)"])


# CEK SATU FOTO (Upload Manual)
with tab1:
    st.header("Tes Prediksi Perorangan")
    uploaded_file = st.file_uploader("Upload 1 file foto...", type=["jpg", "png", "jpeg", "webp"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image = ImageOps.exif_transpose(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Foto Input", use_container_width=True)
        
        with col2:
            st.write("⏳ Menganalisis...")
            try:
                face = mtcnn(image)
                if face is not None:
                    # --- MATH FIX (Float/255) ---
                    face = face.float() / 255.0
                    inp = fixed_image_standardization(face).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = model(inp)
                        probs = torch.nn.functional.softmax(logits, dim=1)
                    
                    conf, idx = torch.max(probs, 1)
                    name = class_names[idx.item()]
                    score = conf.item() * 100
                    
                    if score > 60:
                        st.success(f"### {name}")
                        st.caption(f"Confidence: {score:.2f}%")
                    else:
                        st.warning("Wajah Tidak Dikenal / Ragu")
                        st.write(f"Mirip: {name} ({score:.1f}%)")
                else:
                    st.error("Wajah tidak terdeteksi.")
            except Exception as e:
                st.error(f"Error: {e}")


# UJI DATASET ZIP
with tab2:
    st.header("Hitung Akurasi dari File ZIP")
    
    zip_file = st.file_uploader("Upload File ZIP Dataset", type="zip")
    
    if zip_file and st.button("Mulai Evaluasi"):
        # Folder sementara
        extract_path = "temp_eval"
        if os.path.exists(extract_path): shutil.rmtree(extract_path)
        os.makedirs(extract_path)
        
        # Ekstrak
        with zipfile.ZipFile(zip_file, 'r') as z:
            z.extractall(extract_path)
            
        results = []
        correct = 0
        total = 0
        
        # Cari file gambar
        image_files = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    if '__MACOSX' in root or file.startswith('.'): continue
                    # Nama folder dianggap sebagai label asli
                    true_label = os.path.basename(root)
                    image_files.append((os.path.join(root, file), true_label))
        
        if not image_files:
            st.error("Tidak ada gambar dalam ZIP!")
        else:
            bar = st.progress(0)
            status = st.empty()
            
            for i, (path, true_label) in enumerate(image_files):
                bar.progress((i+1)/len(image_files))
                status.text(f"Memproses: {true_label}...")
                
                pred_label = "Error/No Face"
                is_correct = False
                
                try:
                    img = Image.open(path).convert('RGB')
                    img = ImageOps.exif_transpose(img)
                    face = mtcnn(img)
                    
                    if face is not None:
                        # --- MATH FIX (PENTING) ---
                        face = face.float() / 255.0
                        inp = fixed_image_standardization(face).unsqueeze(0)
                        
                        with torch.no_grad():
                            out = model(inp)
                            probs = torch.nn.functional.softmax(out, dim=1)
                        
                        conf, idx = torch.max(probs, 1)
                        pred_raw = class_names[idx.item()]
                        
                        # Threshold
                        if (conf.item() * 100) > 60:
                            pred_label = pred_raw
                        else:
                            pred_label = "Unknown"
                        
                        # Cek Benar/Salah (Flexible Match)
                        # Benar jika nama folder ada di prediksi atau sebaliknya
                        if pred_label.lower() in true_label.lower() or true_label.lower() in pred_label.lower():
                            is_correct = True
                            correct += 1
                except: pass
                
                total += 1
                results.append({
                    "File": os.path.basename(path),
                    "Asli": true_label,
                    "Prediksi": pred_label,
                    "Status": "✅" if is_correct else "❌"
                })
            
            # HASIL AKHIR
            st.markdown("---")
            accuracy = (correct / total) * 100 if total > 0 else 0
            
            c1, c2 = st.columns(2)
            c1.metric("Total Foto", total)
            c2.metric("Akurasi Dataset", f"{accuracy:.2f}%")
            
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            
            # Bersihkan
            shutil.rmtree(extract_path)
            status.text("Selesai!")