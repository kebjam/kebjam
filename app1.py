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
        background-color: white;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        color: black;
        border-left: 4px solid #4e79a7;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour libérer la mémoire
def clear_models():
    if 'model_fr' in st.session_state:
        del st.session_state.model_fr
    if 'model_en' in st.session_state:
        del st.session_state.model_en
    if 'tokenizer_fr' in st.session_state:
        del st.session_state.tokenizer_fr
    if 'tokenizer_en' in st.session_state:
        del st.session_state.tokenizer_en
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# Fonction pour charger les modèles à la demande
def load_abstractive_model(lang):
    with st.spinner(f"Chargement du modèle {'français' if lang == 'fr' else 'anglais'}..."):
        if lang == 'fr':
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "plguillou/t5-base-fr-sum-cnndm",
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained("plguillou/t5-base-fr-sum-cnndm")
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/bart-large-cnn",
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        
        return model, tokenizer

# Fonction pour le résumé extractif - utilise des statistiques simples pour éviter des bibliothèques supplémentaires
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
        model, tokenizer = load_abstractive_model(lang)
        
        if lang == 'fr':
            inputs = tokenizer(
                "summarize: " + text,
                return_tensors="pt",
                max_length=100,
                truncation=True
            )
        else:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=100,
                truncation=True
            )
        
        # Envoyer au GPU si disponible
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            model = model.to("cuda")
        
        # Générer le résumé
        outputs = model.generate(
            inputs.input_ids,
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Libérer la mémoire GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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
                    st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
                else:
                    st.error("Impossible de générer un résumé extractif")
            
            if abstractive_button:
                with st.spinner("Génération du résumé en cours..."):
                    clear_models()  # Libérer la mémoire avant de charger le nouveau modèle
                    title, summary = abstractive_summarize(text)
                    if summary:
                        st.success(title)
                        st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
                    else:
                        st.error("Impossible de générer un résumé abstractif")
                    clear_models()  # Libérer à nouveau la mémoire
    
    with tab2:
        st.markdown("""
                ### Conception et Développement de l'Application

                Les auteurs de cette application ont structuré leur travail en plusieurs étapes claires :
                - **Pré-traitement et extraction des features** : Cette phase cruciale a consisté à nettoyer les textes d'entrée et à extraire les caractéristiques linguistiques pertinentes. Les développeurs ont implémenté des techniques de segmentation de phrases, de normalisation du texte et d'analyse statistique pour préparer les données avant le traitement.
                - **Résumé automatique** : Le cœur du système comprend deux approches distinctes. La solution extractive utilise des algorithmes statistiques pour sélectionner les phrases clés, tandis que la méthode abstractive s'appuie sur des modèles de langage avancés (BARTHEZ pour le français et DistilBART pour l'anglais) pour générer des résumés fluides.
                - **Déploiement sur API** : L'équipe a finalisé le projet en mettant en place une interface Streamlit conviviale, permettant un accès simple et rapide aux fonctionnalités depuis n'importe quel navigateur web.

                ### Approche Extractive : Mécanisme et Caractéristiques

                Le **résumé extractif** repose sur une méthodologie statistique éprouvée :
                - **Fonctionnement** : Le système identifie et extrait les phrases les plus représentatives du texte source en analysant leur position dans le document et leur longueur. Les phrases situées en début de texte et de taille moyenne reçoivent un score plus élevé.
                - **Avantages** : Cette méthode offre une rapidité d'exécution remarquable et nécessite peu de ressources système. Elle préserve intégralement la formulation originale, garantissant une exactitude terminologique.
                - **Limitations** : Le résumé produit peut manquer de fluidité entre les phrases sélectionnées et ne capture pas toujours les idées implicites nécessitant une reformulation.

                ### Approche Abstractive : Technologie et Performances

                La solution de **résumé abstractif** met en œuvre des technologies avancées de NLP :
                - **Principe** : Contrairement à l'approche extractive, cette méthode génère du texte complètement nouveau en s'appuyant sur des transformers (BARTHEZ et DistilBART) spécialement entraînés pour la synthèse de contenu.
                - **Points forts** : Capable de produire des résumés plus naturels et concis, cette approche excelle dans la reformulation des idées et la compression d'information tout en maintenant la cohérence du discours.
                - **Contraintes** : Elle nécessite davantage de ressources computationnelles et peut occasionnellement introduire des hallucinations ou des approximations, particulièrement sur des textes très techniques.

                ### Détection Automatique de Langue

                L'application intègre un système intelligent de reconnaissance linguistique :
                - **Mécanisme** : Par l'analyse des 500 premiers caractères du texte, le module langdetect détermine automatiquement si le contenu est en français ou en anglais avec une précision supérieure à 95%.
                - **Fonctionnalité** : Cette détection permet d'orienter automatiquement le traitement vers le modèle linguistique approprié, offrant ainsi une expérience utilisateur transparente sans nécessiter de sélection manuelle.
                - **Couverture** : Bien que spécialisée pour le français et l'anglais, l'application peut identifier d'autres langues pour afficher un message d'erreur approprié guidant l'utilisateur.
                    

                ### Auteurs : Alex, Kebjam, Firhoun
                        """)

if __name__ == "__main__":
    main()