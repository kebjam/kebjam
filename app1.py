import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import torch
from transformers import pipeline
import numpy as np

# Configuration pour optimiser l'utilisation de la mémoire
st.set_page_config(
    page_title="Résumé Automatique",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Meta-tag pour viewport mobile et styles CSS
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
    .stTextArea textarea {font-size: 16px !important;}
    .stButton>button {width: 100%;}
    .stAlert {font-size: 14px;}
    .output-text {
        color: black !important;
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #4e79a7;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour libérer la mémoire
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Cache les modèles de manière optimisée
@st.cache_resource
def load_abstractive_fr_model():
    with st.spinner("Chargement du modèle français..."):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "plguillou/t5-base-fr-sum-cnndm",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
        return model, tokenizer

@st.cache_resource
def load_abstractive_en_model():
    with st.spinner("Chargement du modèle anglais..."):
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        return model, tokenizer

@st.cache_resource
def load_extractive_summarizer():
    with st.spinner("Chargement du modèle extractif..."):
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        return summarizer

# Fonction pour le résumé extractif
def extractive_summarize(text, lang):
    # Version simplifiée de l'algorithme extractif basé sur TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    nltk.download('punkt', quiet=True)
    
    # Division du texte en phrases
    sentences = nltk.sent_tokenize(text)
    
    # Si le texte est trop court
    if len(sentences) <= 3:
        return text
    
    # Calcul des scores TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english' if lang == 'en' else None)
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Calcul des scores de chaque phrase
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    
    # Sélection des phrases les plus importantes (30% du texte original)
    num_sentences = max(3, int(len(sentences) * 0.3))
    top_indices = sentence_scores.argsort()[-num_sentences:]
    top_indices = sorted(top_indices)
    
    # Construction du résumé
    summary = " ".join([sentences[i] for i in top_indices])
    return summary

# Fonction pour le résumé abstractif
def abstractive_summarize(text, lang):
    try:
        if lang == 'fr':
            model, tokenizer = load_abstractive_fr_model()
            inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        elif lang == 'en':
            model, tokenizer = load_abstractive_en_model()
            inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            summary = "Langue non supportée (FR/EN seulement)"
            
        # Libérer la mémoire
        clear_memory()
        return summary
    except Exception as e:
        return f"Erreur lors du résumé abstractif: {str(e)}"

# Interface utilisateur
def main():
    st.title("Résumé Automatique (FR/EN)")
    
    # Choix du type de résumé
    col1, col2 = st.columns(2)
    with col1:
        extractive_btn = st.button("Résumé Extractif", use_container_width=True, 
                                  help="Extraits les phrases les plus importantes du texte original")
    with col2:
        abstractive_btn = st.button("Résumé Abstractif", use_container_width=True,
                                   help="Génère un nouveau texte résumant les idées principales")
    
    # Zone de texte
    text = st.text_area("Texte à résumer", height=300)
    
    # Traitement du résumé
    if (extractive_btn or abstractive_btn) and text:
        if len(text) < 50:
            st.warning("Veuillez entrer un texte plus long (minimum 50 caractères)")
        else:
            with st.spinner("Analyse en cours..."):
                try:
                    # Détection de la langue
                    lang = detect(text)
                    
                    if lang not in ['fr', 'en']:
                        st.error("Langue non supportée. Veuillez utiliser du français ou de l'anglais.")
                    else:
                        # Appliquer le type de résumé choisi
                        if extractive_btn:
                            summary = extractive_summarize(text, lang)
                            method = "Extractif"
                        else:  # abstractive_btn
                            summary = abstractive_summarize(text, lang)
                            method = "Abstractif"
                        
                        # Affichage du résultat
                        lang_display = "Français" if lang == 'fr' else "English"
                        st.success(f"**Résumé {method} ({lang_display} détecté):**")
                        
                        st.markdown(f"""
                        <div class="output-text">
                        {summary}
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Erreur: {str(e)}")
                    
                # Libérer la mémoire à la fin
                clear_memory()
    
    # Footer avec instructions
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
    st.caption("**Instructions**: Choisissez d'abord le type de résumé puis entrez votre texte. Le résumé extractif est plus rapide mais moins fluide, tandis que le résumé abstractif est plus naturel mais prend plus de temps.")

# Exécution de l'application
if __name__ == "__main__":
    main()

