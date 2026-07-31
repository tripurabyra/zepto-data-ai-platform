from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

DOC_PATH = Path(__file__).parent / "docs"

CHROMA_PATH = Path(__file__).parent / "vectorstore" / "chroma_db"

documents = []

for file in DOC_PATH.glob("doc_*.txt"):

    print("Loading:", file)

    text = file.read_text(
        encoding="utf-8"
    ).strip()

    if text:
        documents.append({
            "id": file.stem,
            "text": text
        })


print("Total documents loaded:", len(documents))


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to Chroma
client = chromadb.PersistentClient(
    path=str(CHROMA_PATH)
)

collection = client.get_or_create_collection(
    name="zepto_policy_collection"
)


# Remove old empty records
existing = collection.get()

if existing["ids"]:
    collection.delete(
        ids=existing["ids"]
    )


ids = []
texts = []
embeddings = []

for doc in documents:
    ids.append(doc["id"])
    texts.append(doc["text"])
    embeddings.append(
        model.encode(doc["text"]).tolist()
    )


collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings
)


print("Inserted documents:", collection.count())