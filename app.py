import requests
from typing import List, Tuple, Optional, Dict, Any
import json
import streamlit as st
import threading
import time
import math
from queue import Queue

import os
import re
import pickle
import pathlib
import logging
import warnings
import io
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import yake
from wordsegment import load, segment
import ftfy
import pyphen
import pdfplumber
import nltk

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# ===== LOGGING & WARNINGS =====
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== PAGE CONFIG =====
st.set_page_config(
    page_title="AI-DocSearch - Recherche Intelligente",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS =====
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        transition: all 0.3s ease;
    }
    
    .main-title {
        font-size: 3.5em;
        font-weight: 200;
        text-align: center;
        margin: 0.8em 0 0.2em 0;
        letter-spacing: -1px;
        color: #9ffaaa;
        background: linear-gradient(120deg, #e60000, #4d0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: none;
    }
    
    .subtitle {
        font-size: 1.1em;
        color: rgba(255, 255, 255, 0.7);
        text-align: center;
        margin-bottom: 3em;
        font-weight: 300;
        letter-spacing: 0.3px;
    }
    
    .upload-container {
        max-width: 750px;
        margin: 0 auto;
        padding: 1em;
    }
    
    .upload-zone {
        display: none;
    }
    
    .results-counter {
        font-size: 0.9em;
        color: rgba(255, 255, 255, 0.5);
        text-align: center;
        margin: 2em 0;
    }
    
    /* Style général des boutons */
    .stButton button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9em;
    }
    
    .stButton button:hover {
        border-color: rgba(255, 255, 255, 0.2);
        color: white;
    }

    /* Style spécifique pour le bouton Browse files */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(120deg, #ff1a1a, #1a0000);
        border: none;
        color: white;
        font-size: 0.95em;
        padding: 0.8em 1.5em;
        border-radius: 8px;
        font-weight: 500;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(255, 0, 0, 0.3);
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(120deg, #cc0000, #000000);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 0, 0, 0.4);
    }
    
    .block-container {
        max-width: 1000px;
        padding-top: 0;
    }
    
    /* Cacher les éléments non essentiels */
    .stDeployButton, footer, header {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== DEPENDANCY CHECK (auto) =====
@st.cache_data
def check_dependencies():
    missing_deps = []
    try: import PyPDF2
    except ImportError: missing_deps.append("PyPDF2")
    try: import sentence_transformers
    except ImportError: missing_deps.append("sentence-transformers")
    try: import numpy
    except ImportError: missing_deps.append("numpy")
    try: import sklearn
    except ImportError: missing_deps.append("scikit-learn")
    try: import transformers
    except ImportError: missing_deps.append("transformers")
    return missing_deps

missing_deps = check_dependencies()
if missing_deps:
    st.error(f"Dépendances manquantes: {', '.join(missing_deps)}")
    st.info("À installer : pip install " + " ".join(missing_deps))
    st.stop()

# ===== NLTK fallback =====
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
    @st.cache_data
    def download_nltk_data():
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            with st.spinner("Téléchargement des ressources linguistiques..."):
                nltk.download('punkt', quiet=True)
    download_nltk_data()
except ImportError:
    NLTK_AVAILABLE = False
    def sent_tokenize(text):
        # fallback regex découpeur
        return re.split(r'(?<=[.!?])\s+', text)

# ===== WORDSEGMENT INIT =====
load()

# Crée le césureur français une fois
_pyphen_dic = pyphen.Pyphen(lang="fr_FR")

# File d'attente pour les nouveaux documents
document_queue = Queue()

class DocumentHandler(FileSystemEventHandler):
    def __init__(self, watch_path: str, supported_extensions: set):
        self.watch_path = watch_path
        self.supported_extensions = supported_extensions

    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            ext = os.path.splitext(file_path)[1].lower()
            if ext in self.supported_extensions:
                document_queue.put(file_path)
                logger.info(f"Nouveau document détecté: {file_path}")

class AutoProcessor(threading.Thread):
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        super().__init__(daemon=True)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                file_path = document_queue.get(timeout=1)
                self.process_document(file_path)
                document_queue.task_done()
            except Queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erreur traitement automatique: {e}")

    def process_document(self, file_path: str):
        try:
            with open(file_path, 'rb') as file:
                filename = os.path.basename(file_path)
                if filename.endswith('.pdf'):
                    text = extract_text_from_pdf(file)
                elif filename.endswith('.docx'):
                    text = extract_text_from_docx(file)
                elif filename.endswith('.txt'):
                    text = extract_text_from_txt(file)
                else:
                    return

                clean_full = clean_text(text)
                chunks = smart_chunk_text(clean_full, self.chunk_size, self.overlap)
                
                if chunks:
                    st.session_state.all_chunks.extend(chunks)
                    st.session_state.chunk_origins.extend([filename] * len(chunks))
                    st.session_state.document_stats[filename] = {
                        "pages_or_chunks": len(chunks),
                        "characters": sum(len(c) for c in chunks),
                        "words": sum(len(c.split()) for c in chunks)
                    }
                    st.session_state.embeddings = None
                    logger.info(f"Document traité automatiquement: {filename}")
        except Exception as e:
            logger.error(f"Erreur traitement fichier {file_path}: {e}")

def start_document_watcher(watch_path: str):
    supported_extensions = {'.pdf', '.txt', '.docx'}
    event_handler = DocumentHandler(watch_path, supported_extensions)
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=False)
    observer.start()
    return observer

@st.cache_data
def get_suggested_keywords(chunks: List[str], top_k: int = 20) -> List[str]:
    text = " ".join(chunks)
    kw_extractor = yake.KeywordExtractor(lan="fr", n=1, top=top_k)
    keywords = [kw for kw, score in kw_extractor.extract_keywords(text)]
    return keywords

def clean_text(text: str) -> str:
    """Nettoie, décale et segmente même les mots les plus collés."""
    if not text or not isinstance(text, str):
        return ""

    # 0) Fixe encodage & supprime les caractères invisibles
    text = ftfy.fix_text(text)
    text = re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', text)

    # 1) Sépare les SUITES DE MAJ suivies de minuscule : 'RDVdelatech'→'RDV delatech'
    text = re.sub(r'([A-ZÀ-Ÿ]{2,})([a-zà-ÿ])', r'\1 \2', text)

    # 2) Sépare minuscule→MAJ : 'etIntervention'→'et Intervention'
    text = re.sub(r'([a-zà-ÿ])([A-ZÀ-Ÿ])', r'\1 \2', text)

    # 3) Sépare lettre↔chiffre et chiffre↔lettre : 'doc123'→'doc 123', '123doc'→'123 doc'
    text = re.sub(r'([A-Za-z])([0-9])', r'\1 \2', text)
    text = re.sub(r'([0-9])([A-Za-z])', r'\1 \2', text)

    # 4) (ATTENTION) On retire la fusion agressive des lettres pour éviter de coller des mots légitimes
    # text = re.sub(r'([A-Za-zÀ-ÿ])\s+([A-Za-zÀ-ÿ])', r'\1\2', text)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)

    # 5) Gestion de la ponctuation
    text = re.sub(r'\s+([,;:\.\?!%])', r'\1', text)
    text = re.sub(r'([,;:\.\?!%])([^\s])', r'\1 \2', text)

    # 6) Structure puces et sauts de phrase
    text = re.sub(r'•', '\n• ', text)
    text = re.sub(r'\.\s*', '.\n', text)

    # 7) Segmente automatiquement les tokens >10 caractères
    def _seg(m):
        tok = m.group(0)
        parts = segment(tok.lower())
        if parts:
            parts[0] = parts[0].capitalize()
            return " ".join(parts)
        return tok

    text = re.sub(r'\b[^\s]{10,}\b', _seg, text)

    # 8) Nettoyage final des sauts de ligne et espaces multiples
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return text.strip()

def is_mostly_corrupted(text: str) -> bool:
    words = text.split()
    if not words: return True
    single_letters = sum(1 for word in words if len(word) <= 2 and word.isalpha())
    return single_letters / len(words) > 0.3

class EnrichedChunk:
    def __init__(self, text: str, analysis: Dict[str, Any] = None):
        self.text = text
        self.analysis = analysis or {}
        
    def get_enriched_text(self) -> str:
        """Combine le texte original avec l'analyse pour un meilleur embedding"""
        if not self.analysis:
            return self.text
            
        enriched_parts = [self.text]
        
        if self.analysis.get("concepts"):
            enriched_parts.append("Concepts principaux : " + ", ".join(self.analysis["concepts"]))
        if self.analysis.get("keywords"):
            enriched_parts.append("Mots-clés : " + ", ".join(self.analysis["keywords"]))
        if self.analysis.get("context"):
            enriched_parts.append("Contexte : " + self.analysis["context"])
        if self.analysis.get("summary"):
            enriched_parts.append("Résumé : " + self.analysis["summary"])
            
        return " | ".join(enriched_parts)
    
    def __str__(self):
        return self.text

def smart_chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[EnrichedChunk]:
    if not text: return []
    try:
        sentences = sent_tokenize(text)
    except: sentences = re.split(r'[.!?]+', text)
    chunks, current_chunk, seen_chunks = [], "", set()
    
    # Initialise le client Ollama pour l'analyse
    ollama = OllamaClient()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 20: continue
        potential = current_chunk + " " + sentence if current_chunk else sentence
        if len(potential) > chunk_size and current_chunk:
            clean_chunk = clean_text(current_chunk.strip())
            if (len(clean_chunk) > 100 and len(clean_chunk.split()) > 10 and clean_chunk not in seen_chunks and not is_mostly_corrupted(clean_chunk)):
                # Analyse le chunk avec le LLM
                analysis = ollama.analyze_document(clean_chunk)
                chunks.append(EnrichedChunk(clean_chunk, analysis))
                seen_chunks.add(clean_chunk)
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = potential
            
    if current_chunk.strip():
        clean_chunk = clean_text(current_chunk.strip())
        if (len(clean_chunk) > 100 and len(clean_chunk.split()) > 10 and clean_chunk not in seen_chunks and not is_mostly_corrupted(clean_chunk)):
            analysis = ollama.analyze_document(clean_chunk)
            chunks.append(EnrichedChunk(clean_chunk, analysis))
            
    return chunks

def extract_text_from_pdf(file) -> str:
    """Extraction prioritaire avec pdfplumber, fallback PyPDF2."""
    text = ""
    try:
        file.seek(0)
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 10:
                    text += "\n" + page_text + "\n"
        logger.info("PDF lu avec pdfplumber")
    except Exception as e:
        logger.warning(f"pdfplumber a échoué ({e}), fallback PyPDF2")
        file.seek(0)
        reader = PdfReader(io.BytesIO(file.read()))
        for page in reader.pages:
            pt = page.extract_text()
            if pt: text += "\n" + pt + "\n"
    return text

def extract_text_from_txt(file) -> str:
    try:
        file.seek(0)
        content = file.read()
        if isinstance(content, str): return content
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']
        for encoding in encodings:
            try:
                text = content.decode(encoding)
                if len(text.strip()) > 0 and not text.count('\ufffd') > len(text) * 0.1:
                    logger.info(f"TXT décodé avec {encoding}")
                    return text
            except (UnicodeDecodeError, UnicodeError):
                continue
        text = content.decode('utf-8', errors='ignore')
        if len(text.strip()) < 10:
            raise ValueError("Fichier vide ou illisible")
        return text
    except Exception as e:
        logger.error(f"Erreur lecture TXT: {e}")
        raise ValueError(f"Impossible de lire le fichier TXT: {str(e)}")

# ==== MODEL CACHING ====
@st.cache_resource
def load_faiss():
    try:
        import faiss
        return faiss.IndexFlatL2(384)  # 384 is the dimension for all-MiniLM-L6-v2
    except Exception as e:
        st.error(f"Erreur initialisation FAISS: {e}")
        st.stop()

@st.cache_resource
def load_sentence_transformer() -> SentenceTransformer:
    try:
        with st.spinner("🔄 Chargement du modèle d'embeddings..."):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception as e:
        st.error(f"Erreur modèle d'embeddings: {e}")
        st.stop()

@st.cache_resource
def load_summarizer():
    try:
        with st.spinner("🔄 Chargement du modèle de résumé..."):
            return pipeline("summarization", model="t5-small", tokenizer="t5-small", framework="pt")
    except Exception as e:
        st.error(f"Erreur modèle résumé: {e}")
        st.stop()

# ==== EVALUATION & TESTING ====
class TestSet:
    def __init__(self):
        self.queries = []
        self.relevant_passages = []
        self.expected_summaries = []

    def add_test_case(self, query: str, relevant_docs: List[str], expected_summary: str):
        self.queries.append(query)
        self.relevant_passages.append(relevant_docs)
        self.expected_summaries.append(expected_summary)

    def evaluate_search(self, search_func, k: int = 3) -> Dict[str, float]:
        total_precision = 0
        total_recall = 0
        
        for query, relevant in zip(self.queries, self.relevant_passages):
            results = search_func(query, k)
            metrics = calculate_search_metrics(relevant, results, k)
            total_precision += metrics['precision@k']
            total_recall += metrics['recall@k']
        
        n = len(self.queries)
        return {
            'avg_precision@k': total_precision / n if n > 0 else 0,
            'avg_recall@k': total_recall / n if n > 0 else 0
        }

    def evaluate_summarization(self, summarize_func) -> Dict[str, float]:
        total_rouge1 = total_rouge2 = total_rougeL = total_bleu = 0
        
        for query, relevant, expected in zip(self.queries, self.relevant_passages, self.expected_summaries):
            summary = summarize_func(relevant[0] if relevant else "", query)
            metrics = calculate_metrics(summary, expected)
            
            total_rouge1 += metrics.get('rouge1', 0)
            total_rouge2 += metrics.get('rouge2', 0)
            total_rougeL += metrics.get('rougeL', 0)
            total_bleu += metrics.get('bleu', 0)
        
        n = len(self.queries)
        return {
            'avg_rouge1': total_rouge1 / n if n > 0 else 0,
            'avg_rouge2': total_rouge2 / n if n > 0 else 0,
            'avg_rougeL': total_rougeL / n if n > 0 else 0,
            'avg_bleu': total_bleu / n if n > 0 else 0
        }

# ==== SESSION STATE ====
def initialize_session_state():
    defaults = {
        "all_chunks": [],
        "chunk_origins": [],
        "embeddings": None,
        "document_stats": {},
        "processing_errors": [],
        "test_set": TestSet(),
        "auto_processor": None,
        "file_observer": None
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

def reset_session():
    st.session_state.all_chunks = []
    st.session_state.chunk_origins = []
    st.session_state.embeddings = None
    st.session_state.document_stats = {}
    st.session_state.processing_errors = []

def save_session(filepath: str = "session_data.pkl"):
    try:
        data = {
            "all_chunks": st.session_state.all_chunks,
            "chunk_origins": st.session_state.chunk_origins,
            "embeddings": st.session_state.embeddings,
            "document_stats": st.session_state.document_stats
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        logger.error(f"Erreur sauvegarde: {e}")
        return False

def load_session(filepath: str = "session_data.pkl"):
    try:
        if not pathlib.Path(filepath).exists():
            return False
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        st.session_state.all_chunks = data.get("all_chunks", [])
        st.session_state.chunk_origins = data.get("chunk_origins", [])
        st.session_state.embeddings = data.get("embeddings", None)
        st.session_state.document_stats = data.get("document_stats", {})
        return True
    except Exception as e:
        logger.error(f"Erreur chargement: {e}")
        return False

# ==== EXPORT/IMPORT JSON ====
def export_json():
    docs = [
        {
            "document": origin, 
            "text": chunk,
            "chunk_id": i,
            "char_count": len(chunk),
            "word_count": len(chunk.split())
        }
        for i, (chunk, origin) in enumerate(zip(st.session_state.all_chunks, st.session_state.chunk_origins))
    ]
    json_str = json.dumps(docs, ensure_ascii=False, indent=2)
    st.download_button(
        label="📤 Export JSON",
        data=json_str.encode("utf-8"),
        file_name=f"ai_docsearch_export_{len(docs)}_chunks.json",
        mime="application/json",
        use_container_width=True,
        help="Exporte tous les chunks au format JSON"
    )

def process_json_import(uploaded_json):
    try:
        docs = json.load(uploaded_json)
        if not isinstance(docs, list):
            st.error("Format JSON invalide: doit être une liste")
            return
        required_keys = {"document", "text"}
        if not all(required_keys.issubset(doc.keys()) for doc in docs if isinstance(doc, dict)):
            st.error("Format JSON invalide: chaque élément doit contenir 'document' et 'text'")
            return
        st.session_state.all_chunks = [doc["text"] for doc in docs]
        st.session_state.chunk_origins = [doc["document"] for doc in docs]
        st.session_state.embeddings = None
        st.session_state.document_stats = {}
        for doc_name in set(st.session_state.chunk_origins):
            doc_chunks = [chunk for chunk, origin in zip(st.session_state.all_chunks, st.session_state.chunk_origins) if origin == doc_name]
            st.session_state.document_stats[doc_name] = {
                "chunks": len(doc_chunks),
                "characters": sum(len(chunk) for chunk in doc_chunks),
                "words": sum(len(chunk.split()) for chunk in doc_chunks)
            }
        st.success(f"✅ {len(docs)} chunks importés depuis JSON")
        st.rerun()
    except json.JSONDecodeError:
        st.error("Fichier JSON invalide")
    except Exception as e:
        st.error(f"Erreur lors de l'import JSON: {e}")

# ==== GENERATE EMBEDDINGS ====
def generate_embeddings():
    try:
        with st.spinner("🔄 Génération des embeddings enrichis..."):
            model = load_sentence_transformer()
            
            # Utilise le texte enrichi pour les embeddings
            enriched_texts = [
                chunk.get_enriched_text() if isinstance(chunk, EnrichedChunk) else str(chunk)
                for chunk in st.session_state.all_chunks
            ]
            
            if len(enriched_texts) > 100:
                progress_bar = st.progress(0)
                batch_size = 50
                embeddings_list = []
                
                for i in range(0, len(enriched_texts), batch_size):
                    batch = enriched_texts[i:i+batch_size]
                    batch_embeddings = model.encode(batch)
                    embeddings_list.append(batch_embeddings)
                    progress_bar.progress((i + len(batch)) / len(enriched_texts))
                
                st.session_state.embeddings = np.vstack(embeddings_list)
                progress_bar.empty()
            else:
                st.session_state.embeddings = model.encode(enriched_texts)
                
            st.success(f"✅ Embeddings enrichis générés: {st.session_state.embeddings.shape}")
    except Exception as e:
        st.error(f"Erreur lors de la génération des embeddings: {e}")
        logger.error(f"Erreur embeddings: {e}")

def extract_text_from_docx(file) -> str:
    """Extract text from a Word document."""
    try:
        from docx import Document
        file.seek(0)
        doc = Document(io.BytesIO(file.read()))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"Erreur lecture DOCX: {e}")
        raise ValueError(f"Impossible de lire le fichier DOCX: {str(e)}")

def process_uploaded_files(uploaded_files, chunk_size: int, overlap: int):
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_files = len(uploaded_files)
    apercus = []
    for i, file in enumerate(uploaded_files, start=1):
        try:
            status_text.text(f"📖 Traitement de {file.name} ({i}/{total_files})…")
            pages = []
            if file.name.lower().endswith('.pdf'):
                file.seek(0)
                raw = io.BytesIO(file.read())
                try:
                    import pdfplumber
                    with pdfplumber.open(raw) as pdf:
                        for page in pdf.pages:
                            txt = page.extract_text() or ""
                            if len(txt.strip()) > 50:
                                pages.append(txt)
                except Exception:
                    raw.seek(0)
                    reader = PdfReader(raw)
                    for p in reader.pages:
                        txt = p.extract_text() or ""
                        if len(txt.strip()) > 50:
                            pages.append(txt)
            elif file.name.lower().endswith('.docx'):
                raw_text = extract_text_from_docx(file)
                clean_full = clean_text(raw_text)
                pages = smart_chunk_text(clean_full, chunk_size, overlap)
            elif file.name.lower().endswith('.txt'):
                file.seek(0)
                txt_bytes = file.read()
                raw_text = extract_text_from_txt(io.BytesIO(txt_bytes))
                clean_full = clean_text(raw_text)
                pages = smart_chunk_text(clean_full, chunk_size, overlap)
            else:
                st.warning(f"⚠️ Format non supporté : {file.name}")
                continue
            if not pages:
                st.warning(f"⚠️ Aucun contenu valide extrait de {file.name}")
                continue
            st.session_state.all_chunks.extend(pages)
            st.session_state.chunk_origins.extend([file.name] * len(pages))
            st.session_state.document_stats[file.name] = {
                "pages_or_chunks": len(pages),
                "characters": sum(len(p) for p in pages),
                "words": sum(len(p.split()) for p in pages)
            }
            # Ajoute l'aperçu à la liste pour affichage groupé
            apercus.append({
                "filename": file.name,
                "nb_pages": len(pages),
                "preview": pages[0][:500] + ("..." if len(pages[0]) > 500 else "")
            })
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement de {file.name} : {e}")
        finally:
            progress_bar.progress(i / total_files)
    status_text.empty()
    progress_bar.empty()
    st.session_state.embeddings = None

    # Affichage minimal du statut
    if apercus:
        st.markdown(f'''
            <div class="results-counter">
                {len(st.session_state.all_chunks)} pages analysées
            </div>
        ''', unsafe_allow_html=True)

    # Affichage discret des succès
    with st.expander("✨ Voir les détails du traitement", expanded=False):
        for ap in apercus:
            st.markdown('''
                <div class="success-message">
                    ✅ {filename} → {nb_pages} pages importées
                </div>
            '''.format(
                filename=ap['filename'],
                nb_pages=ap['nb_pages']
            ), unsafe_allow_html=True)
            
        # Aperçus dans un sous-expander
        with st.expander("📄 Voir les aperçus des documents", expanded=False):
            for ap in apercus:
                st.markdown(f'''
                    <div style="margin: 1em 0;">
                        <div style="
                            font-size: 0.9em;
                            color: rgba(255, 255, 255, 0.7);
                            margin-bottom: 0.5em;
                        ">
                            Aperçu de <code>{ap['filename']}</code>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
                st.text_area(
                    label="",
                    value=ap["preview"],
                    height=120,
                    disabled=True,
                    key=f"preview_{ap['filename']}"
                )

    # Affichage embeddings si déjà générés
    if st.session_state.embeddings is not None:
        st.success(f"✅ Embeddings générés: {st.session_state.embeddings.shape}")

def search_interface(top_k: int, similarity_threshold: float):
    # Interface minimaliste
    st.markdown('<h1 class="main-title">AI-Powered Document Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Système intelligent de recherche et résumé de documents</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown('''
        <div class="upload-zone">
            <p style="color: rgba(255, 255, 255, 0.8); margin-bottom: 0.5em;">
                Déposez vos documents ici
            </p>
            <p style="color: rgba(255, 255, 255, 0.5); font-size: 0.85em;">
                PDF • TXT • DOCX
            </p>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Suggestions de mots-clés avec style amélioré
    if st.session_state.all_chunks:
        suggested = get_suggested_keywords(st.session_state.all_chunks, top_k=15)
        if suggested:
            st.markdown('''
                <div style="
                    margin: 2em 0;
                    text-align: center;
                    color: rgba(255, 255, 255, 0.7);
                    font-size: 0.9em;
                ">
                    Suggestions : {}
                </div>
            '''.format(
                " ".join([f'<span style="background: rgba(255,255,255,0.1); padding: 4px 8px; border-radius: 4px; margin: 0 4px;">{kw}</span>' for kw in suggested])
            ), unsafe_allow_html=True)

    # Déplacez la création du formulaire en haut niveau
    with st.form(key="search_form"):
        query = st.text_input(
            "Posez votre question ou tapez un mot‑clé :",
            value=st.session_state.get("user_query", ""),
            placeholder="Ex: Comment fonctionne l'IA ?",
            help="Inspirez-vous des suggestions ci-dessus ou saisissez votre propre requête",
            key="user_query"
        )
        
        submitted = st.form_submit_button(label="🔍 Rechercher")
    
    # Vérifiez la soumission en dehors du formulaire
    if submitted:
        final_query = query.strip()
        if not final_query:
            st.warning("Veuillez saisir un mot‑clé ou une question")
        else:
            # Ajoutez un try-catch pour capturer les erreurs
            try:
                perform_search(
                    question=final_query,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                    selected_docs=[],
                    show_full_chunks=True,  # Changez à True pour voir les résultats
                    enable_summary=True,
                )
            except Exception as e:
                st.error(f"Erreur lors de la recherche: {str(e)}")
                logger.error(f"Search error: {str(e)}")


def calculate_metrics(predictions, references) -> Dict[str, float]:
    """Calculate ROUGE and BLEU scores for summarization evaluation."""
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu
    try:
        # ROUGE scores
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(predictions, references)
        
        # BLEU score
        bleu = sentence_bleu([references.split()], predictions.split())
        
        return {
            'rouge1': rouge_scores['rouge1'].fmeasure,
            'rouge2': rouge_scores['rouge2'].fmeasure,
            'rougeL': rouge_scores['rougeL'].fmeasure,
            'bleu': bleu
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}

def calculate_search_metrics(relevant_docs: List[str], retrieved_docs: List[str], k: int) -> Dict[str, float]:
    """Calculate precision@k and recall@k for search evaluation."""
    try:
        retrieved_set = set(retrieved_docs[:k])
        relevant_set = set(relevant_docs)
        
        # Calculate precision@k
        precision_k = len(retrieved_set.intersection(relevant_set)) / k if k > 0 else 0
        
        # Calculate recall@k
        recall_k = len(retrieved_set.intersection(relevant_set)) / len(relevant_set) if relevant_set else 0
        
        return {
            'precision@k': precision_k,
            'recall@k': recall_k
        }
    except Exception as e:
        logger.error(f"Error calculating search metrics: {e}")
        return {}

def calculate_perplexity(text: str) -> float:
    """Calcule la perplexité du texte"""
    try:
        words = text.split()
        if len(words) < 2:
            return 0
        
        # Calcul simple de la perplexité basé sur la distribution des mots
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        prob = 1
        N = len(words)
        for word in words:
            prob *= (word_freq[word] / N)
        
        perplexity = 2 ** (-1/N * sum(math.log2(word_freq[word]/N) for word in words))
        return perplexity
    except Exception as e:
        logger.error(f"Error calculating perplexity: {e}")
        return 0

def perform_search(question: str, top_k: int, similarity_threshold: float,
                   selected_docs: List[str], show_full_chunks: bool, enable_summary: bool) -> List[Tuple[str, float]]:
    # Avant de faire la recherche, vérifiez bien les embeddings
    if st.session_state.embeddings is None or len(st.session_state.embeddings) == 0:
        st.error("Les embeddings ne sont pas encore générés. Veuillez patienter.")
        return []
    
    if len(st.session_state.embeddings.shape) != 2 or st.session_state.embeddings.shape[1] != 384:
        st.error("Format d'embeddings invalide. Veuillez regénérer les embeddings.")
        return []

    try:
        # Crée un conteneur pour le spinner
        with st.spinner("🔍 Analyse de votre requête en cours..."):
            # Initialise le client Ollama
            ollama = OllamaClient()
            
            # Analyse la requête
            analysis = ollama.analyze_query(question)
        if show_full_chunks:
            st.info(f"💡 Requête reformulée : {analysis['reformulation']}")
        
        # Effectue la recherche avec la requête reformulée
        with st.spinner("🔍 Recherche dans les documents en cours..."):
            model = load_sentence_transformer()
            q_emb = model.encode([analysis['reformulation']])
            
            indices = [
            i for i, origin in enumerate(st.session_state.chunk_origins)
            if not selected_docs or origin in selected_docs
        ]
        emb = st.session_state.embeddings[indices]
        chunks = [st.session_state.all_chunks[i] for i in indices]
        origins = [st.session_state.chunk_origins[i] for i in indices]
        sims = cosine_similarity(q_emb, emb)[0]
        scored = sorted(
            [(i, sims[i]) for i in range(len(sims)) if sims[i] >= similarity_threshold],
            key=lambda x: x[1], reverse=True
        )
        
        if not scored:
            if show_full_chunks:
                st.warning("Aucun résultat pertinent.")
            return []

        seen = set()
        results = []
        for idx, score in scored:
            key = chunks[idx][:50]
            if key in seen:
                continue
            seen.add(key)
            results.append((idx, score))
            if len(results) >= top_k:
                break

        # Afficher les résultats seulement si show_full_chunks est True
        if show_full_chunks:
            st.success(f"🎯 {len(results)} résultat(s) affichés sur {len(scored)} trouvés")
            
            for rank, (idx, score) in enumerate(results, start=1):
                orig_idx = indices[idx]
                chunk = chunks[idx]
                origin = origins[idx]
                
                st.markdown(f"## 🏆 Résultat #{rank}")
                
                # Info du document avec style amélioré
                st.markdown(f'''
                    <div style="
                        background: rgba(255, 255, 255, 0.03);
                        padding: 12px 16px;
                        border-radius: 8px;
                        margin: 10px 0;
                        font-size: 0.9em;
                    ">
                        <span style="color: rgba(255, 255, 255, 0.7);">📄 Document:</span>
                        <code style="
                            background: rgba(255, 255, 255, 0.1);
                            padding: 2px 6px;
                            border-radius: 4px;
                            margin: 0 8px;
                        ">{origin}</code>
                        <span style="color: rgba(255, 255, 255, 0.5);">•</span>
                        <span style="color: rgba(255, 255, 255, 0.7); margin-left: 8px;">
                            Position: {orig_idx+1}/{len(st.session_state.all_chunks)}
                        </span>
                    </div>
                ''', unsafe_allow_html=True)

                # Boîte de métriques avec style CSS personnalisé
                metrics_css = """
                <style>
                    .metrics-box {
                        background-color: rgba(255, 255, 255, 0.05);
                        border-radius: 8px;
                        padding: 15px;
                        margin: 10px 0;
                    }
                    .metrics-grid {
                        display: flex;
                        flex-direction: row;
                        flex-wrap: nowrap;
                        gap: 8px;
                        justify-content: center;
                        align-items: stretch;
                        width: 100%;
                    }
                    .metric-item {
                        background-color: rgba(255, 255, 255, 0.08);
                        border-radius: 6px;
                        padding: 8px;
                        flex: 1 1 0;
                        min-width: 0;
                        text-align: center;
                        display: inline-block;
                    }
                    .metric-value {
                        font-size: 1.2em;
                        font-weight: 600;
                        color: #fff;
                        margin: 4px 0;
                    }
                    .metric-label {
                        color: rgba(255, 255, 255, 0.7);
                        font-size: 0.85em;
                        margin-bottom: 2px;
                    }
                    .metric-help {
                        color: rgba(255, 255, 255, 0.5);
                        font-size: 0.75em;
                        margin-top: 4px;
                    }
                </style>
                """
                st.markdown(metrics_css, unsafe_allow_html=True)
                
                st.markdown('''
                    <div class="metrics-box">
                        <div class="metrics-grid">
                ''', unsafe_allow_html=True)
                
                # Toutes les métriques dans une seule rangée
                keywords = set(analysis['keywords'])
                chunk_words = set(chunk.lower().split())
                recall = len(keywords.intersection(chunk_words)) / len(keywords) if keywords else 0
                precision = score
                perplexity = calculate_perplexity(chunk)
                
                st.markdown('''
                    <div style="display: flex; gap: 8px; margin: 10px 0;">
                        <div class="metric-item">
                            <div class="metric-label">Similarité</div>
                            <div class="metric-value">{:.1f}%</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Precision@k</div>
                            <div class="metric-value">{:.2%}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Recall@k</div>
                            <div class="metric-value">{:.2%}</div>
                        </div>
                        <div class="metric-item">
                            <div class="metric-label">Perplexité</div>
                            <div class="metric-value">{:.2f}</div>
                            <div class="metric-help">Plus la perplexité est basse, plus le texte est cohérent</div>
                        </div>
                    </div>
                '''.format(score*100, precision, recall, perplexity), unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)  # Fin de metrics-grid
                st.markdown('</div>', unsafe_allow_html=True)  # Fin de metrics-section
                

                
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("📖 Extrait", expanded=(rank == 1)):
                    # Surligne les mots-clés
                    highlighted = highlight_keywords(chunk, question)
                    st.markdown(highlighted, unsafe_allow_html=True)
                
                if enable_summary:
                    try:
                        # Utilise Ollama pour un résumé intelligent
                        summary = ollama.enhance_summary(chunk, question)
                        st.info(f"📝 Résumé : {summary}")
                        
                        # Calcul des scores ROUGE et BLEU pour le résumé
                        if hasattr(st.session_state, 'last_query_summary'):
                            rouge_scores = calculate_metrics(summary, st.session_state.last_query_summary)
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("ROUGE-1", f"{rouge_scores['rouge1']:.2%}")
                                st.metric("ROUGE-2", f"{rouge_scores['rouge2']:.2%}")
                            with col2:
                                st.metric("ROUGE-L", f"{rouge_scores['rougeL']:.2%}")
                                st.metric("BLEU", f"{rouge_scores['bleu']:.2%}")
                        
                        # Sauvegarde le résumé pour la prochaine comparaison
                        st.session_state.last_query_summary = summary
                        
                    except Exception as e:
                        st.warning(f"Erreur résumé : {e}")
                        
                st.divider()

            # Retourne les résultats
            results_with_chunks = [(chunks[idx], score) for idx, score in results]
            return results_with_chunks

    except Exception as e:
        st.error(f"Erreur lors de la recherche : {e}")

def highlight_keywords(text: str, query: str) -> str:
    """Surligne les mots-clés significatifs de la requête dans le texte."""
    # Liste de mots à ignorer (stop words en français)
    stop_words = {'le', 'la', 'les', 'de', 'des', 'du', 'un', 'une', 'et', 'ou', 'à', 'en', 'dans', 'sur', 'pour', 'par', 'ce', 'ces', 'il', 'elle', 'ils', 'elles', 'je', 'tu', 'nous', 'vous'}
    
    # Extraction des mots significatifs de la requête
    words = [
        re.escape(w.lower()) 
        for w in query.strip().split() 
        if w.lower() not in stop_words and len(w) > 2
    ]
    
    if not words:
        return text
        
    # Construction du pattern avec les mots significatifs
    pattern = r'(' + '|'.join(words) + r')'
    
    # Surlignage avec prise en compte de la casse
    def replace(match):
        return f'<mark>{match.group(0)}</mark>'
    
    return re.sub(pattern, replace, text, flags=re.IGNORECASE)

class OllamaClient:
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.timeout = 30  # Ajoutez un timeout
    
    def generate(self, prompt: str, model: str = "mistral", context: str = "") -> str:
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            logger.info(f"Sending request to Ollama: {payload}")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout  # Ajoutez le timeout
            )
            
            if response.status_code != 200:
                raise ConnectionError(f"Ollama error: {response.text}")
                
            return response.json().get("response", "")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama connection failed: {str(e)}")
            st.error("Le serveur Ollama n'est pas accessible. Vérifiez qu'il est bien démarré.")
            return ""
        
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """Analyse un document pour en extraire les informations clés et le contexte"""
        prompt = f"""Analyse ce texte et retourne un JSON avec :
        - les concepts clés
        - les entités importantes
        - un résumé court
        - des mots-clés pertinents
        - le contexte général

        TEXTE : {text}

        Réponds uniquement avec un JSON de cette forme :
        {{
            "concepts": ["concept1", "concept2", ...],
            "entities": ["entité1", "entité2", ...],
            "summary": "résumé concis",
            "keywords": ["mot-clé1", "mot-clé2", ...],
            "context": "contexte général"
        }}"""
        
        try:
            result = self.generate(prompt)
            result = result.strip()
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                result = result[start:end]
            parsed = json.loads(result)
            return parsed
        except Exception as e:
            logger.error(f"Document analysis error: {e}")
            return {
                "concepts": [],
                "entities": [],
                "summary": text[:200] + "...",
                "keywords": [],
                "context": ""
            }
        
    def generate(self, prompt: str, model: str = "mistral", context: str = "") -> str:
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            logger.info(f"Sending request to Ollama: {payload}")
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload
            )
            
            logger.info(f"Ollama response status: {response.status_code}")
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama error response: {response.text}")
                return ""
        except Exception as e:
            logger.error(f"Ollama connection error: {str(e)}")
            return ""

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyse la requête pour extraire l'intention et les mots-clés"""
        prompt = f"""Tu dois analyser cette requête et retourner un JSON valide qui contient :
        - l'intention principale de l'utilisateur
        - les mots-clés importants
        - une meilleure formulation de la requête

        La requête est : {query}

        Réponds uniquement avec un JSON de cette forme :
        {{
            "intention": "...",
            "keywords": ["...", "..."],
            "reformulation": "..."
        }}"""
        
        try:
            result = self.generate(prompt)
            logger.info(f"Raw Ollama response: {result}")
            
            # Nettoyage plus agressif du JSON
            result = result.strip()
            # Trouve le premier { et le dernier }
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                result = result[start:end]
            
            parsed = json.loads(result)
            logger.info(f"Parsed JSON: {parsed}")
            return parsed
            
        except Exception as e:
            logger.error(f"Query analysis error: {str(e)}")
            return {
                "intention": "search",
                "keywords": query.lower().split(),
                "reformulation": query
            }

    def enhance_summary(self, text: str, query: str) -> str:
        """Génère un résumé amélioré en tenant compte de la requête"""
        prompt = f"""Fais un résumé en français de ce texte qui répond à la requête.

        REQUÊTE : {query}

        TEXTE : {text}

        INSTRUCTIONS :
        - Résumé court et concis
        - En français
        - Concentre-toi sur ce qui répond à la requête
        - Format : texte simple, pas de JSON ni de structure spéciale
        """
        
        try:
            summary = self.generate(prompt)
            return summary if summary else "Résumé non disponible"
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return "Résumé non disponible"
def evaluation_interface():
    st.header("🧪 Évaluation du système")
    
    # Interface de test
    with st.expander("Ajouter un cas de test"):
        test_query = st.text_input("Requête de test")
        test_relevant = st.text_area("Documents pertinents (un par ligne)")
        test_summary = st.text_area("Résumé attendu")
        
        if st.button("Ajouter le cas de test"):
            if test_query and test_relevant and test_summary:
                relevant_docs = [doc.strip() for doc in test_relevant.split('\n') if doc.strip()]
                st.session_state.test_set.add_test_case(test_query, relevant_docs, test_summary)
                st.success("✅ Cas de test ajouté")

    # Lancer l'évaluation
    if st.button("🔍 Évaluer les performances"):
        with st.spinner("Évaluation en cours..."):
            # Évaluation de la recherche
            def search_func(q: str, k: int) -> List[str]:
                results = perform_search(q, k, 0.1, [], False, False)
                if results:
                    return [chunk for chunk, _ in results]
                return []
                
            search_metrics = st.session_state.test_set.evaluate_search(search_func)
            
            # Évaluation des résumés
            summary_metrics = st.session_state.test_set.evaluate_summarization(
                lambda text, query: OllamaClient().enhance_summary(text, query)
            )
            
            # Affichage des résultats dans une grille horizontale
            st.subheader("📊 Métriques d'évaluation")
            metrics_cols = st.columns(6)
            
            # Métriques de recherche
            with metrics_cols[0]:
                st.metric("Precision@k", f"{search_metrics['avg_precision@k']:.2%}")
            with metrics_cols[1]:
                st.metric("Recall@k", f"{search_metrics['avg_recall@k']:.2%}")
            
            # Métriques de résumé
            with metrics_cols[2]:
                st.metric("ROUGE-1", f"{summary_metrics['avg_rouge1']:.2%}")
            with metrics_cols[3]:
                st.metric("ROUGE-2", f"{summary_metrics['avg_rouge2']:.2%}")
            with metrics_cols[4]:
                st.metric("ROUGE-L", f"{summary_metrics['avg_rougeL']:.2%}")
            with metrics_cols[5]:
                st.metric("BLEU", f"{summary_metrics['avg_bleu']:.2%}")

def main():
    initialize_session_state()
    
    # Démarrage du processeur automatique
    if st.session_state.auto_processor is None:
        processor = AutoProcessor()
        processor.start()
        st.session_state.auto_processor = processor
    
    # Démarrage de l'observateur de fichiers
    if st.session_state.file_observer is None:
        watch_path = os.path.dirname(os.path.abspath(__file__))
        observer = start_document_watcher(watch_path)
        st.session_state.file_observer = observer
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Configuration Ollama
        st.subheader("🤖 Configuration Ollama")
        ollama_model = st.selectbox(
            "Modèle Ollama",
            ["mistral", "llama2", "codellama"],
            help="Choisissez le modèle à utiliser pour l'analyse"
        )
        
        chunk_size = st.slider("Taille des chunks", 200, 1200, 500, 50)
        overlap = st.slider("Chevauchement", 0, 300, 50, 25)
        top_k = st.slider("Nombre de résultats", 1, 10, 3)
        similarity_threshold = st.slider("Seuil de similarité (%)", 0, 100, 10, 5) / 100
        st.subheader("📊 Statistiques")
        if st.session_state.all_chunks:
            st.metric("Chunks totaux", len(st.session_state.all_chunks))
            st.metric("Documents", len(set(st.session_state.chunk_origins)))
            if st.session_state.embeddings is not None:
                st.metric("Embeddings", st.session_state.embeddings.shape[0])
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Réinitialiser", use_container_width=True):
            reset_session()
            st.success("Mémoire réinitialisée!")
            st.rerun()
    with col2:
        if st.button("💾 Sauvegarder", use_container_width=True):
            if save_session():
                st.success("Session sauvegardée!")
            else:
                st.error("Erreur lors de la sauvegarde")
    with col3:
        if st.button("📂 Charger", use_container_width=True):
            if load_session():
                st.success("Session restaurée!")
                st.rerun()
            else:
                st.warning("Aucune session trouvée")
    with col4:
        if st.session_state.all_chunks:
            export_json()
    st.header("📁 Import de documents")
    tab1, tab2 = st.tabs(["📄 Fichiers", "📋 Import JSON"])
    with tab1:
        uploaded_files = st.file_uploader(
            "Sélectionnez vos documents (PDF, TXT, DOCX)",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Formats supportés: PDF, TXT, DOCX"
        )
        if uploaded_files:
            process_uploaded_files(uploaded_files, chunk_size, overlap)
    with tab2:
        uploaded_json = st.file_uploader(
            "Importez un fichier JSON exporté",
            type="json",
            help="Fichier JSON au format {document, text}"
        )
        if uploaded_json:
            process_json_import(uploaded_json)
    if st.session_state.all_chunks and st.session_state.embeddings is None:
        generate_embeddings()
    # Interface principale
    tab1, tab2 = st.tabs(["🔍 Recherche", "ℹ️ Documentation"])
    
    with tab1:
        if st.session_state.all_chunks:
            if st.session_state.embeddings is not None:
                search_interface(top_k, similarity_threshold)
            else:
                st.info("⏳ Génération des embeddings en cours...")
        else:
            st.info("📤 Importez des documents pour commencer la recherche")
    
    with tab2:
        st.header("📚 Documentation du système")
        
        st.subheader("Architecture du système")
        st.markdown("""
        Le système AI-DocSearch est composé des éléments suivants :
        
        1. **Ingestion de documents**
        - Formats supportés : PDF, DOCX, TXT
        - Extraction de texte optimisée avec fallback
        - Nettoyage et normalisation automatique
        
        2. **Traitement du texte**
        - Segmentation intelligente en chunks
        - Détection et filtrage du texte corrompu
        - Chevauchement configurable
        
        3. **Embeddings & Recherche**
        - Modèle : all-MiniLM-L6-v2
        - Index vectoriel : FAISS (CPU)
        - Recherche sémantique avec seuil configurable
        
        4. **Génération & Résumé**
        - Modèle : Ollama (mistral/llama2)
        - Analyse intelligente des requêtes
        - Résumés contextuels
        
        5. **Automatisation**
        - Surveillance automatique des nouveaux fichiers
        - File d'attente de traitement
        - Traitement asynchrone
        """)
        
        st.subheader("Performances")
        st.markdown("""
        Le système est optimisé pour les performances CPU :
        
        - Traitement par lots (batch_size=50)
        - Mise en cache des modèles
        - Index vectoriel léger
        - Parallélisation des tâches lourdes
        """)
        
        st.subheader("Métriques d'évaluation")
        st.markdown("""
        Le système utilise plusieurs métriques pour évaluer sa performance :
        
        1. **Recherche**
        - Precision@k
        - Recall@k
        
        2. **Résumé**
        - ROUGE (1, 2, L)
        - BLEU
        """)
        
        st.subheader("Guide d'utilisation")
        st.markdown("""
        1. **Import de documents**
        - Utilisez l'interface de glisser-déposer
        - Formats acceptés : PDF, DOCX, TXT
        - Les documents sont automatiquement traités
        
        2. **Recherche**
        - Entrez votre requête en langage naturel
        - Ajustez les paramètres dans la barre latérale
        - Consultez les suggestions de mots-clés
        
        3. **Évaluation**
        - Ajoutez des cas de test
        - Lancez l'évaluation complète
        - Analysez les métriques
        """)

if __name__ == "__main__":
    main()