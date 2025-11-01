# Mini RAG 

**Models Used:**  
- Embeddings: `text-embedding-3-small` 
- LLM: `gpt-4o-mini` 

**Vector DB:** FAISS (`IndexFlatIP` with cosine via normalization)

## Folder
```
main.py       
rag.py        
requirements.txt
README.md
```

## Dataset
Download the Kaggle CSV (**Wikipedia Movie Plots**) and place it at:
```
data/wiki_movie_plots_deduped.csv
```

## Setup
```bash
Activate you venv first
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...                          
```

## Run
```bash
python main.py
```
Then type questions like:
```
Enter your movie question (or 'exit'): Which film features HAL 9000?
```

## Output (Structured JSON)
```json
{
  "answer": "The movie 2001: A Space Odyssey features HAL 9000.",
  "contexts": ["2001: A Space Odyssey … HAL 9000 becomes antagonistic …"],
  "reasoning": "Searched chunks; top match mentions HAL 9000 and space mission."
}
```
