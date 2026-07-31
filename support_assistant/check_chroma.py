import chromadb

client = chromadb.PersistentClient(
    path="vectorstore/chroma_db"
)

collection = client.get_collection(
    "zepto_policy_collection"
)

result = collection.get()

print("IDs:")
print(result["ids"])

print("\nDocuments:")
for doc in result["documents"]:
    print("----------------")
    print(doc[:200])