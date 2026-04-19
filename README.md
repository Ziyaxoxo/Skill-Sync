## Description
SkillSync is a high-performance Resume Analysis and ATS (Applicant Tracking System) Optimization tool. It utilizes Natural Language Processing (NLP) and Cosine Similarity to evaluate the alignment between a candidate's resume and a specific job description. The system provides a multi-dimensional analysis including keyword extraction, match percentage, and customized interview preparation based on identified skill gaps.

## Core Features
* **ATS Scoring Engine**: Calculates a comprehensive score based on keyword density, formatting checks, and structural integrity.
* **Semantic Analysis**: Employs TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and Cosine Similarity to determine contextual alignment.
* **Skill Extraction**: Automatically parses PDF content to identify technical competencies and cross-references them with Job Description (JD) requirements.
* **Gap Analysis**: Generates dynamic visual tags for matching and missing skills to guide resume tailoring.
* **Predictive Categorization**: Classifies the resume into professional domains (e.g., Data Science, Web Development) using pre-trained ML logic.
* **Interview Intelligence**: Provides a curated set of behavioral and technical questions focused specifically on the user's missing skill sets.

## Technical Architecture
* **Frontend**: Streamlit-based SPA (Single Page Application) with custom CSS injection for optimized UX.
* **NLP Pipeline**: 
    * **Preprocessing**: Custom cleaning functions for text normalization.
    * **Vectorization**: TfidfVectorizer from Scikit-learn for numerical feature representation.
    * **Similarity**: Linear kernel/Cosine Similarity for distance metrics between documents.
* **Extraction**: PyPDF2 or specialized PDF miners for text serialization.
* **Utilities**: Modular backend (`utils.py`) handling scoring logic, skill mapping, and category prediction.

## Installation and Deployment

1. **Clone the repository**:
   git clone https://github.com/Ziyaxoxo/SkillSync.git

2. **Install dependencies**:
   pip install streamlit pandas scikit-learn joblib PyPDF2 spacy

3. **Download Language Models**:
   python -m spacy download en_core_web_sm

## Execution
To launch the local development server:
streamlit run app.py

## Output


<img width="1854" height="957" alt="image" src="https://github.com/user-attachments/assets/46d3fcff-f907-4143-8640-4428a9a4852c" />
<br>
<img width="1846" height="954" alt="image" src="https://github.com/user-attachments/assets/7cf23b23-8a8a-4546-94d0-61d187e32ed6" />
<br>
<img width="1854" height="953" alt="image" src="https://github.com/user-attachments/assets/bde3cfb9-c6c4-48ec-af5d-0466972ccd8c" />


## Usage Workflow
1. **Ingestion**: Upload a PDF resume through the dropzone.
2. **Contextualization**: Paste the target Job Description into the text area.
3. **Inference**: Click "Analyze Match Compatibility" to trigger the NLP pipeline.
4. **Optimization**: Review the "Missing Skills" and "ATS Breakdown" to refine the resume.

## Dependencies
* **Streamlit**: Application framework.
* **Scikit-learn**: Machine learning and similarity metrics.
* **TF-IDF Vectorizer**: Feature extraction.
* **Plotly**: Visual analytics (where applicable).
