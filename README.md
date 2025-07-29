# AI-DocSearch 🔍

Une application de recherche sémantique intelligente pour vos documents, construite avec Streamlit et des modèles d'IA avancés.

Voici le lien de la vidéo de présentation de la plateforme: https://youtu.be/59USzSNciLU .

## 🌟 Fonctionnalités

- **Import Multi-format** : Support pour PDF, DOCX et TXT
- **Recherche Sémantique** : Utilisation d'embeddings pour une recherche contextuelle précise
- **Analyse Intelligente** : Analyse automatique du contenu avec Ollama
- **Résumé Automatique** : Génération de résumés contextuels
- **Interface Intuitive** : Design moderne et réactif
- **Métriques en Temps Réel** : Scores de similarité, précision et perplexité
- **Gestion de Session** : Sauvegarde et restauration de l'état
- **Export/Import JSON** : Partage et réutilisation des données

## 🚀 Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/ID18030104/Hackaton02.git
cd Hackaton02
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Installez et démarrez Ollama (requis pour l'analyse et les résumés) :
- Suivez les instructions sur [ollama.ai](https://ollama.ai)
- Assurez-vous que le serveur Ollama est en cours d'exécution

## 📋 Prérequis

- Python 3.8+
- Streamlit
- Sentence Transformers
- PyPDF2
- python-docx
- NLTK
- FAISS
- Ollama

## 🎯 Utilisation

1. Lancez l'application :
```bash
streamlit run app.py
```

2. Importez vos documents via l'interface de glisser-déposer

3. Attendez la génération des embeddings

4. Commencez à rechercher !

## 🔧 Configuration

Ajustez les paramètres dans la barre latérale :
- Taille des chunks (200-1200 caractères)
- Chevauchement des chunks (0-300 caractères)
- Nombre de résultats (1-10)
- Seuil de similarité (0-100%)
- Modèle Ollama (mistral, llama2, codellama)

## 📊 Métriques

L'application fournit plusieurs métriques pour évaluer la qualité des résultats :
- **Similarité** : Score de correspondance sémantique
- **Precision@k** : Précision des résultats retournés
- **Recall@k** : Couverture des résultats pertinents
- **Perplexité** : Mesure de la cohérence du texte

## 🛠️ Architecture

1. **Ingestion de Documents**
   - Extraction de texte multi-format
   - Nettoyage et normalisation
   - Segmentation intelligente

2. **Traitement du Texte**
   - Chunking adaptatif
   - Détection de corruption
   - Enrichissement sémantique

3. **Recherche et Analyse**
   - Embeddings avec all-MiniLM-L6-v2
   - Index vectoriel FAISS
   - Analyse sémantique Ollama

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📝 License

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.

## 👥 Auteurs

- ID18030104

## 🙏 Remerciements

- [Streamlit](https://streamlit.io)
- [Sentence-Transformers](https://www.sbert.net)
- [Ollama](https://ollama.ai)
