🩺 QDoctor
A Specialized AI Tool for Mental Health & Medical Guidelines in Kenya

QDoctor is a system designed to provide accurate, evidence-based answers to mental health and clinical queries. It leverages the WHO mhGAP and WHO ICD guidelines,NICE Guidelines, the Kenya MOH Mental Health Guidelines, and Kenya GovernmentLaws on Clinical protocols to ensure that responses are grounded in verified medical literature.

Key Features
Semantic Search: Uses vector embeddings to understand the intent behind medical queries rather than just keyword matching.

Kenya-Specific Knowledge: Deeply indexed with Kenyan mental health laws and national clinical guidelines.

Safety First: Features a built-in Decision Engine to filter non-medical queries and a Hallucination Checker to ensure answers match the source documents.

Optimized Performance: Implements QCache for lightning-fast retrieval on frequent questions.

📂 Project Structure
Plaintext
QDoctor-2.0/
├── config/             # System settings (API keys, Top-K, thresholds)
├── core/               # The "Brain" (Orchestrator, Retriever, Generator)
├── evaluation/         # Safety layers (Validator, Hallucination Checker)
├── QBrain/             # Your medical knowledge base (PDFs/Guidelines)
├── storage/            # FAISS index and persistent vector data
├── synthesis/          # Prompt engineering and response formatting
├── main.py             # Entry point for the backend server
├── setup.py            # Data ingestion and index building script
└── requirements.txt    # Project dependencies
Tech Stack
LLM: Groq (Llama-3-70b/8b)

Vector Store: FAISS (Facebook AI Similarity Search)

Embeddings: Sentence-Transformers / HuggingFace

Orchestration: Python


Installation & Setup
Clone the repository:

Bash
git clone https://github.com/duotkuerduot/Qdoctor-2.0.git
cd Qdoctor-2.0
Install dependencies:

Bash
pip install -r requirements.txt
Set up environment variables: Create a .env file in the root directory:

Code snippet
GROQ_API_KEY=your_api_key_here
Initialize the Knowledge Base: Run the setup script to index your PDF documents in QBrain/:

Bash
python setup.py
Run the tool:

Bash
python main.py
Safety & Disclaimer
QDoctor is an information retrieval tool designed to support clinical decision-making and provide legal clarity on mental health acts. It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a physician or other qualified health provider with any questions regarding a medical condition.
