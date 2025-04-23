import streamlit as st
from langdetect import detect
import torch
from transformers import pipeline
import nltk

# Télécharger les ressources NLTK nécessaires (si pas déjà fait)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configuration pour mobiles et optimisation mémoire
st.set_page_config(
    page_title="Résumé Automatique Mobile",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Style mobile et optimisations
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<style>
    .stTextArea textarea {font-size: 16px !important;}
    .stButton>button {width: 100%;}
    .stAlert {font-size: 14px;}
    .summary-box {
        padding: 15px;
        border-radius: 5px;
        background-color: #f0f2f6;
        border-left: 4px solid #4e79a7;
        margin-top: 10px;
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour libérer la mémoire
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Cache le modèle simplifié de manière optimisée
@st.cache_resource
def load_simple_model():
    with st.spinner("Chargement du modèle léger..."):
        # Utilisation du pipeline de résumé de Hugging Face avec un modèle plus léger
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn-samsum",  # Modèle plus léger pour le résumé de conversations
            device=0 if torch.cuda.is_available() else -1,
        )
        return summarizer

# Fonction pour le résumé extractif - utilise des statistiques simples
def extractive_summarize(text, ratio=0.3):
    # Détection de la langue
    try:
        lang = detect(text)
    except:
        return "Erreur de détection de langue", ""
    
    if lang not in ['fr', 'en']:
        return "Langue non supportée (FR/EN seulement)", ""
    
    # Split text into sentences using NLTK
    sentences = nltk.sent_tokenize(text, language='french' if lang == 'fr' else 'english')
    
    if not sentences:
        return "Pas assez de texte pour résumer", ""
    
    # Récupérer les stopwords selon la langue
    try:
        stopwords = set(nltk.corpus.stopwords.words('french' if lang == 'fr' else 'english'))
    except:
        stopwords = set()
    
    # Calculer la fréquence des mots (hors stopwords)
    word_freq = {}
    for sentence in sentences:
        for word in nltk.word_tokenize(sentence.lower(), language='french' if lang == 'fr' else 'english'):
            if word not in stopwords and word.isalnum():
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # Normaliser les fréquences
    max_freq = max(word_freq.values()) if word_freq else 1
    for word in word_freq:
        word_freq[word] = word_freq[word] / max_freq
    
    # Score sentences based on word frequency
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        # Score based on position (beginning sentences are usually more important)
        position_score = 1.0 if i < len(sentences) // 3 else 0.5
        
        # Score based on word frequency
        word_score = sum([word_freq.get(word.lower(), 0) 
                         for word in nltk.word_tokenize(sentence, language='french' if lang == 'fr' else 'english')
                         if word.isalnum()])
        
        # Calculate average word score for the sentence
        if len(sentence.split()) > 0:
            word_score = word_score / len(sentence.split())
        else:
            word_score = 0
        
        # Combine scores
        total_score = position_score * 0.3 + word_score * 0.7
        sentence_scores[sentence] = total_score
    
    # Select top sentences
    num_sentences = max(1, int(len(sentences) * ratio))
    top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
    
    # Reorder sentences based on original position
    summary_sentences = []
    for sentence in sentences:
        if sentence in [s[0] for s in top_sentences]:
            summary_sentences.append(sentence)
    
    # Join sentences
    summary = " ".join(summary_sentences)
    
    return f"**Résumé Extractif ({'Français' if lang == 'fr' else 'Anglais'} détecté):**", summary

# Fonction simplifiée pour le résumé abstractif
def simplified_abstractive_summarize(text, max_length=150):
    try:
        lang = detect(text)
    except:
        return "Erreur de détection de langue", ""
    
    if lang not in ['fr', 'en']:
        return "Langue non supportée (FR/EN seulement)", ""
    
    try:
        # Utiliser un modèle plus léger via pipeline
        summarizer = load_simple_model()
        
        # Limiter la taille d'entrée pour éviter les problèmes de mémoire
        max_input_length = 1024
        if len(text) > max_input_length:
            text = text[:max_input_length]
        
        # Générer le résumé
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
        
        # Libérer la mémoire
        clear_memory()
        
        return f"**Résumé Abstractif ({'Français' if lang == 'fr' else 'Anglais'} détecté):**", summary
        
    except Exception as e:
        return f"Erreur: {str(e)}", ""

# Interface utilisateur principale
def main():
    st.title("Résumé Automatique (FR/EN)")
    
    # Préparation des onglets
    tab1, tab2 = st.tabs(["Résumé de texte", "À propos"])
    
    with tab1:
        text = st.text_area("Texte à résumer", height=250)
        
        # Les deux boutons côte à côte
        col1, col2 = st.columns(2)
        with col1:
            extractive_button = st.button("Résumé Extractif", type="secondary", use_container_width=True)
        with col2:
            abstractive_button = st.button("Résumé Abstractif", type="primary", use_container_width=True)
        
        # Affichage d'informations sur les boutons
        with st.expander("Quelle méthode choisir?"):
            st.markdown("""
            - **Résumé Extractif**: Sélectionne les phrases les plus importantes du texte original. Plus rapide, moins de ressources.
            - **Résumé Abstractif**: Génère un nouveau texte qui capture l'essentiel du contenu. Utilise un modèle optimisé pour mobiles.
            """)
        
        if not text:
            st.info("Veuillez entrer un texte à résumer")
        elif len(text) < 50:
            st.warning("Veuillez entrer un texte plus long (minimum 50 caractères)")
        else:
            # Traitement des boutons
            if extractive_button:
                title, summary = extractive_summarize(text)
                if summary:
                    st.success(title)
                    st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                else:
                    st.error("Impossible de générer un résumé extractif")
            
            if abstractive_button:
                with st.spinner("Génération du résumé abstractif en cours..."):
                    title, summary = simplified_abstractive_summarize(text)
                    if summary:
                        st.success(title)
                        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)
                    else:
                        st.error("Impossible de générer un résumé abstractif")
                    # Libérer la mémoire
                    clear_memory()
    
    with tab2:
        st.markdown("""
        ### À propos de l'application
        
        Cette application propose deux méthodes de résumé automatique:
        
        **Résumé Extractif**:
        - Sélectionne les phrases importantes du texte original
        - Utilise NLTK et des algorithmes de scoring avancés
        - Rapide et léger en ressources
        
        **Résumé Abstractif**:
        - Utilise un modèle d'IA optimisé pour les appareils mobiles
        - Génère un résumé cohérent sans copier le texte mot pour mot
        - Équilibre entre qualité et performance
        
        L'application détecte automatiquement si votre texte est en français ou en anglais.
        
        *Développé avec Streamlit, NLTK et Hugging Face Transformers*
        """)

if __name__ == "__main__":
    main()