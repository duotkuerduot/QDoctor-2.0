---
title: QDoctor AI
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_url: https://qdoctor-ai.vercel.app/
---

# QDoctor AI: Clinical Evidence Synthesis for Kenya

[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://qdoctor-ai.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

QDoctor AI is a specialized clinical decision support system designed to provide high-fidelity, evidence-based responses to mental health and medical queries. By grounding responses in the Global Mental Health Resources i.e **WHO, NICE Guidelines and the Kenya MOH Mental Health Resources among others**, QDoctor ensures clinical accuracy for healthcare providers and legal clarity on Kenyan mental health protocols.

## Experience the Platform
The production version of QDoctor AI, featuring a professional clinical interface with structured synthesis and numbered references, is available at:
👉 **[https://qdoctor-ai.vercel.app/](https://qdoctor-ai.vercel.app/)**

---

## How It Works: The QDoctor Pipeline

QDoctor utilizes a multi-stage RAG (Retrieval-Augmented Generation) pipeline designed to eliminate hallucinations and maximize relevance.

### 1. Intent & Reasoning Engine
Before any search occurs, the system utilizes a **Llama-3-8b** model—fine-tuned for high-reasoning tasks—to analyze the user's intent. 
- **Query Decomposition:** Breaks complex clinical questions into sub-components.
- **Safety Gatekeeping:** Automatically filters non-medical or harmful queries before they enter the retrieval stage.

### 2. Hybrid Search Architecture (Semantic + Keyword)
QDoctor employs a **Weighted Ensemble Retrieval (RRF)** system to ensure no critical guideline is missed:
- **Semantic Search:** Powered primarily by the highly efficient `all-MiniLM-L6-v2` for rapid vector retrieval in constrained environments, with NCBI’s 768-dimensional `MedCPT` integrated as the secondary option for maximum clinical precision.
- **Keyword Search (BM25):** Ensures precise matching for specific drug names, legislative acts, and technical medical terms found in Kenyan MOH guidelines.
- **Reciprocal Rank Fusion (RRF):** Merges both results to deliver the most relevant chunks to the generator.

### 3. OpenEvidence-Style Synthesis
The generation layer is optimized for clinical readability:
- **Structural Subheadings:** Responses are organized by bold subheadings (e.g., **Management**, **Diagnostic Criteria**).
- **Numbered References:** Every claim is followed by an inline citation [1], mapped to a strict **### References** list at the footer (cleanly parsed without exposing local system paths).

### 4. Hallucination & Validation Layer
Every response undergoes a rigorous **Hallucination Check**. The system compares the generated answer against the retrieved context chunks. If the answer cannot be 100% verified against the source text, the system defaults to a "Cannot Verify" safety state rather than inventing information.

---

## 📂 Project Structure

```bash
QDOCTOR-2.0/
├── config/
├── core/
├── evaluation/
├── QBrain/             # Knowledge Base
├── storage/
├── synthesis/
├── main.py
└── requirements.txt
```


## Installation & Setup

1. Clone & Install

```bash
git clone https://github.com/duotkuerduot/QDoctor-2.0.git
cd QDoctor-2.0
```
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables: Create a .env file in the root directory:

```bash
GROQ_API_KEY=your_groq_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
```

4. Initialize the Knowledge Base:
QDoctor's backend is self-initializing. Simply run the main file. If the vector indices do not exist, the system will automatically parse, clean, and embed your documents from /QBrain before starting the server.

```bash
python main.py
```

## Tech Stack

-**Core Model**: Llama-3-8b (Optimized for 70b-level clinical reasoning)
-**Embeddings**: all-MiniLM-L6-v2 (Fast/Default) & ncbi/MedCPT-Query-Encoder (High-Fidelity Medical)
-**Vector DB**: FAISS (Facebook AI Similarity Search) + BM25
-**Frontend**: Next.js (Deployed on Vercel)
-**Auth & Database**: Supabase

## Safety & Disclaimer
QDoctor is an AI assistant designed to support clinical decision-making. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions regarding a medical condition.