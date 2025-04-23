import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect
import torch

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

# Cache les modèles de manière optimisée
@st.cache_resource
def load_model_fr():
    with st.spinner("Chargement du modèle français..."):
        model_fr = AutoModelForSeq2SeqLM.from_pretrained(
            "plguillou/t5-base-fr-sum-cnndm",
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        tokenizer_fr = AutoTokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
        return model_fr, tokenizer_fr

@st.cache_resource
def load_model_en():
    with st.spinner("Chargement du modèle anglais..."):
        model_en = AutoModelForSeq2SeqLM.from_pretrained(
            "facebook/bart-large-cnn",
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        tokenizer_en = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        return model_en, tokenizer_en

# Fonction pour le résumé extractif - utilise des statistiques simples
def extractive_summarize(text, ratio=0.3):
    # Détection de la langue
    try:
        lang = detect(text)
    except:
        return "Erreur de détection de langue", ""
    
    if lang not in ['fr', 'en']:
        return "Langue non supportée (FR/EN seulement)", ""
    
    # Split text into sentences
    if lang == 'fr':
        sentence_separators = ['.', '!', '?', '...', '\n']
    else:
        sentence_separators = ['.', '!', '?', '...', '\n']
    
    sentences = []
    current_sentence = ""
    
    for char in text:
        current_sentence += char
        if any(current_sentence.strip().endswith(sep) for sep in sentence_separators):
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            current_sentence = ""
    
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    
    if not sentences:
        return "Pas assez de texte pour résumer", ""
    
    # Score sentences by length and position
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        # Score based on position (beginning sentences are usually more important)
        position_score = 1.0 if i < len(sentences) // 3 else 0.5
        
        # Score based on length (ignore very short sentences)
        length_score = min(1.0, len(sentence) / 100)
        
        # Combine scores
        total_score = position_score * 0.6 + length_score * 0.4
        
        scored_sentences.append((sentence, total_score))
    
    # Sort by score and select top sentences
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    num_sentences = max(1, int(len(sentences) * ratio))
    summary_sentences = [s[0] for s in scored_sentences[:num_sentences]]
    
    # Reorder sentences based on original position
    original_order = {}
    for i, sentence in enumerate(sentences):
        if sentence in [s[0] for s in scored_sentences[:num_sentences]]:
            original_order[sentence] = i
    
    summary_sentences.sort(key=lambda s: original_order.get(s, 0))
    
    # Join sentences
    summary = " ".join(summary_sentences)
    
    return f"**Résumé Extractif ({'Français' if lang == 'fr' else 'Anglais'} détecté):**", summary

# Fonction pour le résumé abstractif
def abstractive_summarize(text):
    try:
        lang = detect(text)
    except:
        return "Erreur de détection de langue", ""
    
    if lang not in ['fr', 'en']:
        return "Langue non supportée (FR/EN seulement)", ""
    
    try:
        if lang == 'fr':
            model_fr, tokenizer_fr = load_model_fr()
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
        else:  # 'en'
            model_en, tokenizer_en = load_model_en()
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
            - **Résumé Abstractif**: Génère un nouveau texte qui capture l'essentiel du contenu. Plus proche d'un résumé humain mais utilise plus de ressources.
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
                    title, summary = abstractive_summarize(text)
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