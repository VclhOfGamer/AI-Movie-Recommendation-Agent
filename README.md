# AI-Movie-Recommendation-Agent
An AI model can suggest movie via user history data. This model uses Groq - a LLM providing a huge number of free tokens. 
AI Agent can give recommendation base on your history and the reason. Moreover, it can talk about how other people have similar or unsimilar taste think about this film.

To setup this model, follow these steps:
1. Get a free Groq API key
→ https://console.groq.com → Sign up → API Keys → Create key  
# No credit card needed. Free tier: 14,400 requests/day.

2. Backend
# create virtual environment
python -m venv venv
venv\Scripts\activate
# Install dependencies (sentence-transformers will download ~80MB model on first run)
pip install -r requirements.txt
# Configure environment
$env:GROQ_API_KEY="...." enter your key here
$env:DATA_DIR="...\data\ml-latest-small-filtered" path to dataset
# Build ChromaDB index — run ONCE (~5-15 min, CPU only, no API cost)
python -m data.indexer
# Start API server
uvicorn main:app --reload --port 8000

3. Frontend
npm install
npm run dev

# Sample queries
- "What should I watch tonight?"
- "I want a dark psychological thriller with a twist"
- "What do people with similar taste to mine think about Pulp Fiction?"
- "Why do you think I'd like that?"
- "I liked Toy Story but I'm tired of animated movies — what else?"
