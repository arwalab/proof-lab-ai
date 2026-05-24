# Proof Lab AI

A sophisticated Streamlit application for bakery R&D, recipe evaluation, SOP generation, and technical knowledge retrieval.

## Features
- **Ask Knowledge Base**: RAG-powered chat for baking science and techniques.
- **SOP Creator**: Generate professional, staff-ready bakery SOPs.
- **Batch Tracker**: Log and analyze baking batches and process parameters.
- **Vision Analyzer**: Upload images of baked goods for AI-powered visual diagnosis.
- **Recipe R&D Generator**: Invent highly original bakery concepts based on flavor and texture goals.
- **Recipe Evaluator**: Critically evaluate test recipes using technical bakery knowledge.

## Local Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/proof-lab-ai.git
   cd proof-lab-ai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Click "New app" and select this repository.
4. Set the main file path to `app.py`.
5. Click on "Advanced settings" and add your secrets (API keys) in the format:
   ```toml
   OPENAI_API_KEY="your_openai_api_key"
   LLAMA_CLOUD_API_KEY="your_llama_cloud_api_key"
   ```
6. Click "Deploy"!
