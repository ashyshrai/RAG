# query.py
import chromadb
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Connect to Chroma
client = chromadb.HttpClient(host="localhost", port=8000)

print("****** List of Collections *******")
print(client.list_collections())
print("**********************************")


# Retrieve the same collection
collection = client.get_collection("test_documents_v2")
query="i want to learn coding"
# Query the collection
results = collection.query(
    query_texts=[query],
    n_results=10
)

#scoring the docs on rereanker model 
docs=results['documents'][0]
pairs = [(query, doc) for doc in docs]
scores = reranker.predict(pairs)
scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

for ans in docs:
    print('-- ',ans)

# Show top reranked results
print("****** RERANKED RESULTS *******")
for doc, score in scored_docs[:5]:
    print(f"[{score:.3f}] {doc}")
