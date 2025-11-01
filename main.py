import json
from rag import MovieRAG

def loop():
    rag = MovieRAG()
    print(" Building index from 300 movie plots (Title + Plot)\n")
    rag.build()
    print("Ready! Ask movie questions. Type 'exit' to quit.\n")

    while True:
        q = input("Enter your movie question (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            print("Goodbye!")
            break
        result = rag.ask(q)
        print("\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n")

if __name__ == "__main__":
    loop()
