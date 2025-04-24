import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, CamembertTokenizer, CamembertModel, RobertaTokenizer, RobertaModel
from langdetect import detect
import torch
# import gradio as gr  # Commenté car non utilisé
from transformers import pipeline
from functools import lru_cache

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

# Fonction pour le résumé extractif 
# ==================== MODÈLES ====================
@lru_cache(maxsize=None)
def load_camembert_model():
    """Charge le modèle CamemBERT pour le français"""
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertModel.from_pretrained("camembert-base")
    return tokenizer, model

@lru_cache(maxsize=None)
def load_roberta_model():
    """Charge le modèle RoBERTa pour l'anglais"""
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    return tokenizer, model

# ==================== DéTECTION LANGUE ====================
def detect_language(text):
    """Détecte la langue avec gestion des erreurs"""
    try:
        lang = detect(text[:1000])  # Analyse les 1000 premiers caractères
        return lang if lang in ['fr', 'en'] else 'en'  # Fallback à l'anglais
    except:
        return 'en'  # Fallback si échec détection

# ==================== FONCTION PRINCIPALE ====================
def extractive_summarize(text, ratio=0.3):
    # Détection de langue
    lang = detect_language(text)
    
    if lang == 'fr':
        return _french_summary(text, ratio)
    else:
        return _english_summary(text, ratio)

# ==================== VERSION FRANÇAISE ====================
def _french_summary(text, ratio):
    try:
        tokenizer, model = load_camembert_model()
        
        # Découpage des phrases françaises
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in ['.', '!', '?', '…', '\n']:
                if len(current.strip()) > 3:
                    sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        
        if len(sentences) < 2:
            return "Texte trop court", text[:200] + "..."

        # Encodage avec CamemBERT
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        doc_embedding = outputs.last_hidden_state.mean(dim=1)

        # Scoring des phrases
        scored = []
        for i, sent in enumerate(sentences):
            sent_inputs = tokenizer(sent, return_tensors="pt", truncation=True)
            with torch.no_grad():
                sent_embedding = model(**sent_inputs).last_hidden_state.mean(dim=1)
            score = torch.cosine_similarity(doc_embedding, sent_embedding).item()
            scored.append((sent, score, i))
        
        # Sélection
        top = sorted(scored, key=lambda x: -x[1])[:max(2, int(len(sentences)*ratio))]
        summary = ' '.join([s[0] for s in sorted(top, key=lambda x: x[2])])
        
        # Métriques
        orig = len(text.split())
        summ = len(summary.split())
        reduction = f"{orig}→{summ} mots (-{int(100*(1-summ/orig))}%)"
        
        return f"Résumé Extractif (FR) - {reduction}", summary
        
    except Exception as e:
        return f"Erreur FR: {str(e)}", ""

# ==================== VERSION ANGLAISE ====================
def _english_summary(text, ratio):
    try:
        tokenizer, model = load_roberta_model()
        
        # Découpage phrases anglaises
        sentences = []
        current = ""
        for char in text:
            current += char
            if char in ['.', '!', '?', '\n']:
                if len(current.strip().split()) > 3:
                    sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        if len(sentences) < 2:
            return "Text too short", text[:200] + "..."

        # Encodage RoBERTa
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        doc_embedding = outputs.last_hidden_state.mean(dim=1)

        # Scoring avancé
        scored = []
        for i, sent in enumerate(sentences):
            sent_inputs = tokenizer(sent, return_tensors="pt", truncation=True)
            with torch.no_grad():
                sent_embedding = model(**sent_inputs).last_hidden_state.mean(dim=1)
            similarity = torch.cosine_similarity(doc_embedding, sent_embedding).item()
            score = 0.6*similarity + 0.2*(0.9**i) + 0.2*min(1.0, len(sent.split())/15)
            scored.append((sent, score, i))
        
        # Sélection
        top = sorted(scored, key=lambda x: -x[1])[:max(2, int(len(sentences)*ratio))]
        summary = ' '.join([s[0] for s in sorted(top, key=lambda x: x[2])])
        
        # Métriques
        orig = len(text.split())
        summ = len(summary.split())
        metrics = f"Original: {orig} words , Summary: {summ}, words (-{int(100*(1 - orig/summ))}%)"
      
        
        return f"Extractive Summary (EN)\n{metrics}", summary

    except Exception as e:
        return f"EN Error: {str(e)}", ""



# Fonction pour le resume abstractif
def abstractive_summarize(text):
    try:
        lang = detect(text)
    except:
        return "Erreur de detection de langue", ""
    
    if lang not in ['fr', 'en']:
        return "Langue non supportee (FR/EN seulement)", ""
    
    try:
        model, tokenizer = load_abstractive_model(lang)
        
        # Tokenize sans troncation pour calculer la longueur totale
        full_tokens = tokenizer(text, return_tensors="pt", truncation=False)
        input_length = len(full_tokens['input_ids'][0])
        dynamic_max_length = input_length // 4  # 1/4 de la longueur totale

        if lang == 'fr':
            inputs = tokenizer(
                "summarize: " + text,
                return_tensors="pt",
                max_length=dynamic_max_length,  # Longueur dynamique
                truncation=True
            )
        else:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=dynamic_max_length,  # Longueur dynamique
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
        
        # Calcul de la réduction
        orig_length = len(text.split())
        summ_length = len(summary.split())
        reduction = f"Réduction: {orig_length} mots → {summ_length} mots (-{int(100*(1-summ_length/orig_length))}%)"
        
        # Libérer la mémoire GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  
            
        return f"**Résumé Abstractif ({'Français' if lang == 'fr' else 'Anglais'} détecté) - {reduction}**", summary
        
    except Exception as e:
        return f"Erreur: {str(e)}", ""

# Interface utilisateur principale
def main():
    st.title("Résumé Automatique (FR/EN)")
    
    # Préparation des onglets
    tab1, tab2, tab3 = st.tabs(["Résumé de texte", "Présentation des Modèles", "À propos"])
    
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
                # Pour un texte français
                title, summary = extractive_summarize(text)

                # Pour un texte anglais
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
            ### **Modèles Extractifs**

            #### **1. CamemBERT-base (Français)**
            **Fonctionnement** :  
            CamemBERT est une version française du modèle BERT, pré-entraîné sur un large corpus francophone. Pour le résumé extractif, il analyse les embeddings des phrases et sélectionne celles qui sont les plus représentatives du document en calculant leur similarité avec l'embedding moyen du texte.  

            #### **2. RoBERTa-base (Anglais)**  
            **Fonctionnement** :  
            RoBERTa est une version améliorée de BERT, optimisée pour l'anglais. Comme CamemBERT, il est utilisé ici pour extraire les phrases clés via des embeddings, mais avec des pondérations adaptées aux structures anglaises (ex: prise en compte des phrases longues et de leur position).  

            ### **Modèles Abstractifs**  

            #### **1. `plguillou/t5-base-fr-sum-cnndm` (Français)**  
            **Fonctionnement** :  
            Ce modèle T5 (Text-To-Text Transfer Transformer) est fine-tuné pour le résumé abstractif de textes français. Il reformule le contenu en générant des phrases nouvelles qui condensent l’information.  

            #### **2. `facebook/bart-large-cnn` (Anglais)**  
            **Fonctionnement** : BART (Bidirectional and Auto-Regressive Transformer) est un modèle hybride optimisé pour le résumé abstractif en anglais. Il combine une compréhension bidirectionnelle du contexte avec une génération auto-régressive. 
            
            ---

            **Recommandation** :  
            - **Extractif** : Pour une analyse précise sans déformation (médical, juridique).  
            - **Abstractif** : Pour des résumés grand public ou des textes longs nécessitant synthèse.  

                                    
            """)
    with tab3:
        st.markdown("""
                ### Conception et Développement de l'Application

                Les auteurs de cette application ont structuré leur travail en plusieurs étapes claires :
                - **Résumé automatique** : Le cœur du système comprend deux approches distinctes. La solution extractive utilise des algorithmes statistiques pour sélectionner les phrases clés, tandis que la méthode abstractive s'appuie sur des modèles de langage avancés (BARTHEZ pour le français et DistilBART pour l'anglais) pour générer des résumés fluides.
                - **Déploiement sur API** : L'équipe a finalisé le projet en mettant en place une interface Streamlit conviviale, permettant un accès simple et rapide aux fonctionnalités depuis n'importe quel navigateur web.


                ### Détection Automatique de Langue

                L'application intègre un système intelligent de reconnaissance linguistique :
                - **Mécanisme** : Par l'analyse des 500 premiers caractères du texte, le module langdetect détermine automatiquement si le contenu est en français ou en anglais.
                - **Fonctionnalité** : Cette détection permet d'orienter automatiquement le traitement vers le modèle linguistique approprié.
                    

                #### Auteurs : 
                ###### Alex Trésor MEZANVOU KAMDOUN, 
                ###### Kebjam JACKSON et 
                ###### Ahmed Firhoun Oumarou SOULEYE
                """)    

if __name__ == "__main__":
    main()
