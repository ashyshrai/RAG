# query.py
import chromadb

# Connect to Chroma
client = chromadb.HttpClient(host="localhost", port=8000)

print("****** List of Collections *******")
print(client.list_collections())
print("**********************************")


# Retrieve the same collection
collection = client.get_collection("test_documents")

# Query the collection
results = collection.query(
    query_texts=["i want to learn coding"],
    n_results=5
)



print(results['documents'][0])
