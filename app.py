import streamlit as st
from PyPDF2 import PdfReader

st.title("Recherche intelligente : upload, extraction et découpage du texte")

uploaded_files = st.file_uploader(
    "Importez vos documents (PDF ou TXT)", 
    type=["pdf", "txt"], 
    accept_multiple_files=True,
    key="main_file_uploader"
)

chunk_size = 500

all_chunks = []
chunk_origins = []

if uploaded_files:
    for file in uploaded_files:
        st.write(f"**Nom du fichier :** {file.name}")
        if file.name.endswith('.pdf'):
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        elif file.name.endswith('.txt'):
            text = file.read().decode('utf-8', errors='ignore')
        else:
            text = ""

        # Découpage du texte
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        all_chunks.extend(chunks)
        chunk_origins.extend([file.name] * len(chunks))

        st.write(f"**Nombre de chunks générés pour ce fichier :** {len(chunks)}")
        st.write(f"**Premier chunk :** {chunks[0][:400]}{'...' if len(chunks[0]) > 400 else ''}")
        st.divider()

    st.success(f"Nombre total de chunks importés : {len(all_chunks)}")
else:
    st.info("Uploade au moins un fichier PDF ou TXT pour commencer.")


from sentence_transformers import SentenceTransformer
import numpy as np

# Après la génération de all_chunks
if all_chunks:
    st.subheader("Vectorisation des chunks (embeddings)")
    # 1. Load le modèle (1 seule fois)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # 2. Embedding
    embeddings = model.encode(all_chunks)
    st.write(f"Embeddings shape : {embeddings.shape}")  # (nb_chunks, 384)
else:
    embeddings = np.array([])
