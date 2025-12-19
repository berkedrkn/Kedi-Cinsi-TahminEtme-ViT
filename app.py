import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Kedi Cinsi Tahmini", layout="centered")

st.title("🐱 Kedi Cinsi Tahmini")
st.write("Bir kedi fotoğrafı yükleyin, yapay zeka hangi cins olduğunu Türkçe olarak söylesin.")


MODEL_INPUT_SIZE = (224, 224)


BREED_TR = {
    "abyssinian": "Habeş Kedisi",
    "bengal": "Bengal Kedisi",
    "birman": "Birman (Kutsal Burma)",
    "bombay": "Bombay Kedisi",
    "british_shorthair": "Britanya Kısa Tüylü",
    "egyptian_mau": "Mısır Mau",
    "maine_coon": "Maine Coon",
    "persian": "İran Kedisi",
    "ragdoll": "Ragdoll",
    "russian_blue": "Rus Mavisi",
    "siamese": "Siyam Kedisi",
    "sphynx": "Sfenks (Tüysüz)",
    "burmese": "Burmese (Birmanya Kedisi)"
}


CAT_BREEDS = list(BREED_TR.keys())

def get_turkish_name(label: str) -> str:
    
    clean_label = str(label).lower().replace(" ", "_")
   
    return BREED_TR.get(clean_label, clean_label.replace("_", " ").title())

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="weileluc/vit-base-oxford-iiit-pets")

model = load_model()

if "history" not in st.session_state:
    st.session_state.history = []


uploaded_file = st.file_uploader("📤 Fotoğraf yükle", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    original_image = Image.open(uploaded_file).convert("RGB")
    resized_image = original_image.resize(MODEL_INPUT_SIZE)

    
    with st.spinner("Yapay zeka analiz ediyor..."):
        preds = model(resized_image, top_k=3)
        
       
        best_pred = preds[0]
        best_label_raw = best_pred["label"].lower().replace(" ", "_")
        best_tr_name = get_turkish_name(best_label_raw)
        best_score = best_pred["score"]

    
    if best_label_raw not in CAT_BREEDS:
        st.warning(f"⚠️ Bu bir kedi olmayabilir. En yakın tahmin: **{best_tr_name}**")
    else:
        st.success(f"### Tahmin: {best_tr_name} (%{best_score*100:.2f})")
    
    st.image(original_image, use_container_width=True)

    
    with st.expander("🛠️ Teknik Detay: Görüntü Ön İşleme (Resize)"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption=f"Orijinal Boyut: {original_image.size}", use_container_width=True)
        with col2:
            st.image(resized_image, caption=f"Model Girişi: {resized_image.size}", use_container_width=True)

    
    st.subheader("📊 Diğer Olası Cinsler")
    for p in preds:
        tr_name = get_turkish_name(p['label'])
        score = p['score'] * 100
        st.write(f"- **{tr_name}**: %{score:.2f}")
        st.progress(p['score'])

    
    st.session_state.history.append({
        "image": original_image, 
        "label": best_tr_name, 
        "score": best_score
    })


if st.session_state.history:
    st.divider()
    st.subheader("🗂 Geçmiş")
    for item in st.session_state.history[::-1][:5]:
        h_col1, h_col2 = st.columns([1, 4])
        with h_col1:
            st.image(item["image"], width=80)
        with h_col2:
            st.write(f"**{item['label']}** (%{item['score']*100:.1f})")

    if st.button("🧹 Geçmişi Temizle"):
        st.session_state.history = []
        st.rerun()
