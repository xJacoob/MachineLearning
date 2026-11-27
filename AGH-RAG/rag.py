from ingest import Chunk, EMBED_MODEL, CHROMA_DIR, COLLECTION_NAME
import ollama
import chromadb

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(COLLECTION_NAME)
CHAT_MODEL = "qwen2.5:7b"
SYSTEM_PROMPT = """
<role>
You are an AGH students assistant who response on question from college documents. 
You will be given question and several chunks. Based on provided information give an answer
</role>

<rules>
1. Answer only from provided documents, do not create your own response.
2. Cite sources ONLY from the fragment headers. Each fragment starts with a header like "[1] regulamin.pdf, s. 19" — copy the file name and page number exactly from there.
3. If none of the fragments of provided document contains the answer on provided question, give an unequivocal answer:  [Nie znalazłem tej informacji w dokumentach].
4. Answer only in polish language, briefly 1-4 sentences but if questions require bigger answer like List all points then list them all.
</rules>
"""

class SearchResult(Chunk):
    distance: float

def search(question: str, k_top: int = 3) -> list[SearchResult]:
    result_list: list[SearchResult] = []
    embed_question = ollama.embed(model=EMBED_MODEL, input=question)
    result = collection.query(embed_question.embeddings, n_results=k_top)

    documents = result["documents"]; assert documents is not None
    metadatas = result["metadatas"]; assert metadatas is not None
    distances = result["distances"]; assert distances is not None

    for text, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
        source = metadata['source']; assert isinstance(source, str)
        page = metadata['page']; assert isinstance(page, int)

        search_result = SearchResult(text=text, source=source, page=page, distance=distance)
        result_list.append(search_result)

    return result_list

def build_context(search_result: list[SearchResult]) -> str:
    context = []
    for idx, item in enumerate(search_result, start=1):
        context.append(f"[{idx}] {item['source']}, s. {item['page']}\n{item['text']}")

    return "\n\n".join(context)

def chat_answer(question: str, context: str) -> str:
    PROMPT = f"Fragments:\n{context}\nQuestion:\n{question}"
    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": PROMPT},
        ],
        options={"temperature": 0}
    )

    return response.message.content

if __name__ == "__main__":
    while True:
        QUESTION = input("Zadaj pytanie:\n")
        if QUESTION == "quit":
            break
        answer = search(QUESTION, k_top=3)
        context = build_context(answer)
        chat_response = chat_answer(QUESTION, context)
        print(chat_response)

        print(f"Źródła (fragmenty przekazane modelowi):")

        for idx, fragment in enumerate(answer, start=1):
            print(f"[{idx}] {fragment['source']}, s. {fragment['page']}")