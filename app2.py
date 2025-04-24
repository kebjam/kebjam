import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, CamembertTokenizer, CamembertModel, AutoTokenizer, AutoModel
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

@lru_cache(maxsize=None)
def load_extractive_model():
    """Charge un modèle multilingue pour traitement FR/EN"""
    model_name = "bert-base-multilingual-cased"  # Modèle supportant FR + EN
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def extractive_summarize(text, ratio=0.3, lang='auto'):
    try:
        # Détection automatique de la langue si non spécifiée
        if lang == 'auto':
            from langdetect import detect
            lang = detect(text[:500])  # Analyse les 500 premiers caractères
        
        # Chargement du modèle
        tokenizer, model = load_extractive_model()
        
        # Découpage en phrases adapté aux deux langues
        sentences = []
        current = ""
        sentence_endings = {'.', '!', '?', '…', '\n'}
        
        for char in text:
            current += char
            if char in sentence_endings:
                if len(current.strip()) > 3:  # Ignore les phrases trop courtes
                    sentences.append(current.strip())
                current = ""
        
        if current.strip():
            sentences.append(current.strip())

        if len(sentences) < 2:
            return "Texte trop court", text[:200] + ("..." if len(text) > 200 else "")
        
        # Encodage du texte complet
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        doc_embedding = outputs.last_hidden_state.mean(dim=1)  # Embedding moyen du document
        
        # Calcul des scores des phrases
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            # Encodage de la phrase
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            sent_embedding = outputs.last_hidden_state.mean(dim=1)
            
            # Score = similarité cosinus + pondération par position
            similarity = torch.cosine_similarity(doc_embedding, sent_embedding).item()
            position_score = 1.0 - (i / len(sentences))  # Les premières phrases comptent plus
            total_score = (0.7 * similarity) + (0.3 * position_score)
            
            sentence_scores.append((sentence, total_score, i))
        
        # Sélection et réorganisation des phrases
        sentence_scores.sort(key=lambda x: -x[1])  # Tri par score décroissant
        selected = sorted(
            sentence_scores[:max(2, int(len(sentences) * ratio))],
            key=lambda x: x[2]  # Tri par position originale
        )
        summary = ' '.join([s[0] for s in selected])
        
        # Métriques
        orig_words = len(text.split())
        summ_words = len(summary.split())
        reduction = int(100 * (1 - summ_words / orig_words)) if orig_words > 0 else 0
        
        # Formatage du résultat
        lang_label = "FR" if lang == 'fr' else "EN"
        title = f"Résumé Extractif ({lang_label}) - {orig_words}→{summ_words} mots (-{reduction}%)"
        
        return title, summary
        
    except Exception as e:
        return f"Erreur: {str(e)}", ""









"""
@lru_cache(maxsize=None)
def load_camembert_model():
    #""""Charge le modèle extractif""""
    tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
    model = CamembertModel.from_pretrained("camembert-base")
    return tokenizer, model

def extractive_summarize(text, ratio=0.3):
    try:
        # Chargement du modèle
        tokenizer, model = load_camembert_model()
        
        # Tokenization et encodage
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Récupération des embeddings
        embeddings = outputs.last_hidden_state[0]
        
        # Découpage en phrases (version française)
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
        
        # Calcul des scores avec CamemBERT
        sentence_scores = []
        for i, sent in enumerate(sentences):
            # Embedding moyen de la phrase
            sent_inputs = tokenizer(sent, return_tensors="pt", truncation=True)
            with torch.no_grad():
                sent_embedding = model(**sent_inputs).last_hidden_state.mean(dim=1)
            
            # Similarité avec l'embedding global
            score = torch.cosine_similarity(embeddings.mean(dim=0), sent_embedding)
            sentence_scores.append((sent, score.item(), i))
        
        # Sélection des phrases
        sentence_scores.sort(key=lambda x: -x[1])
        selected = sorted(sentence_scores[:int(len(sentences)*ratio)], key=lambda x: x[2])
        summary = ' '.join([s[0] for s in selected])
        
        # Métriques
        orig_len = len(text.split())
        summ_len = len(summary.split())
        reduction = int(100*(1 - summ_len/orig_len)) if orig_len > 0 else 0
        
        return f"Résumé Extractive (FR) - {orig_len}→{summ_len} mots (-{reduction}%)", summary
        
    except Exception as e:
        return f"Erreur: {str(e)}", ""
"""

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
            ### **Modèle Extractif avec CamemBERT**  
            **Fonctionnement** :  
            Cette approche utilise CamemBERT, un modèle de langue français et anglais pré-entraînés, pour générer des embeddings (représentations vectorielles) des phrases. Elle sélectionne ensuite les phrases les plus importantes en calculant leur similarité avec l'embedding moyen du texte, sans générer de nouveau contenu.  

            **Avantages** :  
            - **Préservation exacte** du texte original, crucial pour les documents juridiques ou techniques.  
            - **Performant en français** grâce à l'optimisation de CamemBERT pour cette langue.  
            - **Moins gourmand en ressources** que les modèles abstractifs (pas de génération de texte).  

            **Inconvénients** :  
            - **Non spécialisé** : CamemBERT n'est pas conçu spécifiquement pour le résumé.  
            - **Qualité variable** : dépend fortement de la structure du texte original.  
            - **Limité aux langues supportées** (français optimal, anglais moins performant).  

            ---

            ### **Modèle Abstractif Français : `plguillou/t5-base-fr-sum-cnndm`**  
            **Fonctionnement** :  
            Ce modèle T5 (Text-To-Text Transfer Transformer) est spécialement fine-tuné pour le résumé abstractif en français. Il reformule le contenu pour produire un résumé fluide et naturel, comme le ferait un humain.  

            **Avantages** :  
            - **Résultats très naturels** grâce à son entraînement sur des données journalistiques (CNN/Daily Mail).  
            - **Spécialisé français** : meilleure qualité que les modèles multilingues.  
            - **Capacité à synthétiser** des idées complexes.  

            **Inconvénients** :  
            - **Risque de paraphrase inexacte** (hallucinations).  
            - **Taille importante** (~1 Go), nécessitant une GPU pour des performances optimales.  
            - **Moins adapté** aux textes techniques ou spécialisés.  

            ---

            ### **Modèle Abstractif Anglais : `facebook/bart-large-cnn`**  
            **Fonctionnement** :  
            BART (Bidirectional and Auto-Regressive Transformer) est un modèle state-of-the-art optimisé pour le résumé abstractif en anglais. Il combine compréhension contextuelle et génération de texte.  

            **Avantages** :  
            - **Excellente qualité** sur les textes journalistiques et narratifs.  
            - **Architecture bidirectionnelle** qui capture mieux le contexte que les modèles séquentiels.  
            - **Bon équilibre** entre extraction d'idées et reformulation.  

            **Inconvénients** :  
            - **Très gourmand en ressources** (1.5 Go de RAM minimum).  
            - **Nécessite des prompts adaptés** pour les textes hors domaine journalistique.  
            - **Spécialisé anglais** : performances médiocres en français.  

            ---

            ### **Comparaison Clé**  
            | Critère               | Extractif (CamemBERT) | Abstractif (T5/BART) |  
            |-----------------------|----------------------|----------------------|  
            | **Préservation texte** | ✅ Exacte            | ❌ Reformulé         |  
            | **Fluidité**          | ❌ Original          | ✅ Naturelle         |  
            | **Besoins GPU**       | Faibles              | Élevés               |  
            | **Cas d'usage**       | Textes techniques    | Résultats grand public |  

            **Recommandation** :  
            - Pour **l'extraction fidèle** (rapports, articles scientifiques) : approche CamemBERT.  
            - Pour **des résumés fluides** (articles, contenus web) : modèles T5/BART selon la langue.  

            Ces modèles couvrent l'essentiel des besoins en traitement automatique de texte multilingue, chacun avec ses forces spécifiques.
            """)
    with tab3:
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
                    

                ###### Auteurs : 
                ###### Alex Trésor MEZANVOU KAMFOUN, 
                ###### Kebjam JACKSON et 
                ###### Ahmed Firhoun Oumarou SOULEYE
                """)    

if __name__ == "__main__":
    main()
