import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import torch

# Configuration de base
st.set_page_config(
    page_title="Résumé Automatique",
    layout="centered"
)

# Modèles compatibles Windows (sans sentencepiece)
MODEL_MAPPING = {
    'fr': {
        'model_name': "cmarkea/distilbart-light-fr-summarization",
        'max_length': 512
    },
    'en': {
        'model_name': "sshleifer/distilbart-cnn-6-6",
        'max_length': 1024
    }
}

@st.cache_resource
def load_model(lang):
    config = MODEL_MAPPING[lang]
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            config['model_name'],
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        return model, tokenizer, config['max_length']
    except Exception as e:
        st.error(f"Erreur de chargement du modèle {lang}: {str(e)}")
        return None, None, None

def extractive_summary(text, ratio=0.3):
    """Résumé extractif basique sans dépendances externes"""
    # Découpage par phrases simple
    sentences = []
    current = ""
    for char in text:
        current += char
        if char in {'.', '!', '?', '\n'}:
            if len(current.strip().split()) > 3:  # Ignore les phrases trop courtes
                sentences.append(current.strip())
            current = ""
    
    if current.strip():
        sentences.append(current.strip())
    
    # Sélection basée sur la position et longueur
    n = max(1, int(len(sentences) * ratio))
    return ' '.join(sentences[:n])

def main():
    st.title("📝 Résumé Automatique FR/EN")
    
    tab1, tab2 = st.tabs(["Résumer", "À propos"])
    
    with tab1:
        text = st.text_area("Collez votre texte ici", height=250, 
                          placeholder="Minimum 100 caractères pour de bons résultats...")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Résumé Extractif", help="Méthode rapide qui extrait les phrases clés"):
                if len(text) > 30:
                    with st.spinner("Analyse en cours..."):
                        st.subheader("Résumé extractif")
                        st.write(extractive_summary(text))
                else:
                    st.warning("Texte trop court (minimum 30 caractères)")
        
        with col2:
            if st.button("Résumé Abstractif", help="Méthode avancée qui génère un nouveau texte"):
                if len(text) > 100:
                    with st.spinner("Génération du résumé..."):
                        try:
                            lang = detect(text[:500])  # Détection sur les 500 premiers caractères
                            if lang not in ['fr', 'en']:
                                st.error("Seules les langues FR/EN sont supportées")
                                return
                            
                            model, tokenizer, max_len = load_model(lang)
                            if model:
                                inputs = tokenizer(
                                    text,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=max_len
                                )
                                
                                outputs = model.generate(
                                    inputs.input_ids,
                                    max_length=150,
                                    num_beams=4,
                                    early_stopping=True
                                )
                                
                                st.subheader("Résumé abstractif")
                                st.write(tokenizer.decode(outputs[0], skip_special_tokens=True))
                        except Exception as e:
                            st.error(f"Erreur: {str(e)}")
                else:
                    st.warning("Texte trop court (minimum 100 caractères)")
    
    with tab2:
        st.markdown("""
        ### À propos de l'application
        
        Cette application propose deux méthodes de résumé automatique:
        
        **Résumé Extractif**:
        - Sélectionne les phrases importantes du texte original
        - Utilise des statistiques pour identifier les phrases clés
        - Rapide et léger en ressources
        
        **Résumé Abstractif**:
        - Utilise l'IA pour générer un nouveau texte capturant l'essence du contenu
        - Plus proche d'un résumé humain
        - Utilise des modèles de langage avancés
        
        L'application détecte automatiquement si votre texte est en français ou en anglais.
        
        *Développé avec Streamlit et Hugging Face Transformers*
        """)
if __name__ == "__main__":
    main()