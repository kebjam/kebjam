import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import torch

# Configuration pour mobiles
st.set_page_config(
    page_title="Résumé Automatique Mobile",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Meta-tag pour viewport mobile
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
    .stTextArea textarea {font-size: 16px !important;}
    .stButton>button {width: 100%;}
    .stAlert {font-size: 14px;}
</style>
""", unsafe_allow_html=True)

# Cache les modèles
@st.cache_resource
def load_models():
    with st.spinner("Chargement des modèles..."):
        # Modèle français
        model_fr = AutoModelForSeq2SeqLM.from_pretrained(
            "plguillou/t5-base-fr-sum-cnndm",
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer_fr = AutoTokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
        
        # Modèle anglais
        model_en = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn",
            device_map="auto",
            torch_dtype=torch.float16
        )
        tokenizer_en = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        return model_fr, tokenizer_fr, model_en, tokenizer_en

# Interface utilisateur
def main():
    st.title("Résumé Automatique (FR/EN)")
    st.markdown("""
    <style>
    .stTextArea textarea {font-size: 16px !important;}
    </style>
    """, unsafe_allow_html=True)

    text = st.text_area("Texte à résumer", height=300)

    if st.button("Générer le résumé", type="primary"):
        if not text or len(text) < 50:
            st.warning("Veuillez entrer un texte plus long")
        else:
            with st.spinner("Analyse en cours..."):
                try:
                    lang = detect(text)
                    model_fr, tokenizer_fr, model_en, tokenizer_en = load_models()

                    if lang == 'fr':
                        inputs = tokenizer_fr(
                            "summarize: " + text,
                            return_tensors="pt",
                            max_length=1024,
                            truncation=True
                        ).to(model_fr.device)
                        outputs = model_fr.generate(
                            inputs.input_ids,
                            max_length=150,
                            num_beams=4,
                            early_stopping=True
                        )
                        summary = tokenizer_fr.decode(outputs[0], skip_special_tokens=True)
                        st.success("**Résumé (Français détecté):**")

                    elif lang == 'en':
                        inputs = tokenizer_en(
                            text,
                            return_tensors="pt",
                            max_length=1024,
                            truncation=True
                        ).to(model_en.device)
                        outputs = model_en.generate(
                            inputs.input_ids,
                            max_length=150,
                            num_beams=4,
                            early_stopping=True
                        )
                        summary = tokenizer_en.decode(outputs[0], skip_special_tokens=True)
                        st.success("**Summary (English detected):**")

                    else:
                        st.error("Langue non supportée (FR/EN seulement)")
                        summary = ""

                    if summary:
                        st.markdown(f"""
                        <div style='
                            padding: 15px;
                            border-radius: 5px;
                            background-color: black;
                            border-left: 4px solid #4e79a7;
                        '>
                        {summary}
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Erreur: {str(e)}")

if __name__ == "__main__":
    main()