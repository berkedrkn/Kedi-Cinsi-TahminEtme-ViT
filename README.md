# 🐾 CatVision: Vision Transformer ile Akıllı Sınıflandırma Sistemi

Bu proje, görüntü işleme ve derin öğrenme tekniklerini kullanarak kedi ırklarını yüksek doğrulukla tespit eden bir sistemdir.

## 🛠️ Projenin Yol Haritası (Nasıl Çalışır?)
Uygulama, ham bir görüntüyü alıp sonuca dönüştürmek için şu adımları izler:

1. **Görüntü Girişi:** Kullanıcı yerel cihazından bir kedi fotoğrafı yükler.
2. **Ön İşleme (Preprocessing):** Görüntü, ViT modelinin beklediği standartlara getirilmek üzere Pillow kütüphanesi ile `224x224` piksel boyutuna normalize edilir.
3. **Özellik Çıkarımı:** Vision Transformer mimarisi, resmi $16 \times 16$ boyutundaki "patch"lere (yama) bölerek analiz eder.
4. **Sınıflandırma:** HuggingFace üzerindeki önceden eğitilmiş model, çıkarılan özellikleri işleyerek en yüksek olasılığa sahip ilk 3 kedi cinsini belirler.
5. **Görselleştirme:** Sonuçlar, Streamlit arayüzünde hem görsel hem de olasılık çubukları ile kullanıcıya sunulur.

## ✨ Öne Çıkan Özellikler
* **Gerçek Zamanlı Analiz:** Görüntü yüklendiği anda model saniyeler içinde yanıt verir.
* **Top-3 Olasılık:** Sadece tek bir sonuç değil, modelin şüphelendiği en yakın 3 cinsi gösterir.
* **Analiz Geçmişi:** Oturum boyunca yapılan tüm tahminleri hafızada tutar ve "Geçmişi Temizle" özelliği sunar.
* **Şeffaf İşleme:** Modelin resmi nasıl gördüğünü (resizing işlemi) arayüzde gösterir.

## 🧠 Teknik Mimari
- **Model:** `vit-base-patch16-224` (Vision Transformer).
- **Girdi Boyutu:** $224 \times 224$ (RGB).
- **Yazılım Dili:** Python 3.12.
- **Arayüz:** Streamlit Framework.



## 🚀 Kurulum ve Çalıştırma

1. **Bağımlılıkları Yükleyin:**
   ```bash
   py -m pip install -r requirements.txt
2. **Uygulamayı başlatın::**   
   streamlit run app.py

   🛠️ Kullanılan Teknolojiler
Python: Geliştirme dili.
PyTorch & Transformers: Model yükleme ve çıkarım (inference).
Streamlit: Web tabanlı kullanıcı arayüzü.
Pillow (PIL): Görüntü ön işleme ve format yönetimi.

📂 Proje Dizini
app.py: Arayüz mantığı ve model entegrasyonu.
requirements.txt: Bağımlılık listesi.
README.md: Teknik dokümantasyon.