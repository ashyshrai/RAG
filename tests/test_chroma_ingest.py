import chromadb
import hashlib
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever,EnsembleRetriever
from langchain_community.vectorstores import Chroma


# Connect to Chroma
client = chromadb.HttpClient(host="localhost", port=8000)

# Create or get a collection
collection = client.get_or_create_collection("test_documents_v2")

topics = {
    "Science": [
        "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
        "Einstein's theory of relativity revolutionized physics.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "The periodic table organizes elements by atomic number.",
        "DNA carries genetic information in all living organisms."
    ],
    "Technology": [
        "Python is one of the most popular programming languages for AI.",
        "Blockchain enables decentralized and secure digital transactions.",
        "Cloud computing allows on-demand access to computing resources.",
        "Quantum computing uses qubits to perform complex computations.",
        "The Internet of Things connects everyday devices to the internet."
    ],
    "AI": [
        "Neural networks mimic the structure of the human brain.",
        "Large language models are trained on massive text datasets.",
        "Reinforcement learning trains agents through trial and error.",
        "Computer vision enables machines to interpret images and videos.",
        "Natural language processing helps computers understand human language."
    ],
    "Business": [
        "Supply chain management ensures smooth flow of goods and services.",
        "Financial markets react to macroeconomic indicators and policies.",
        "Startups often rely on venture capital for early growth.",
        "Brand identity helps companies differentiate from competitors.",
        "Leadership drives team performance and innovation."
    ],
    "Art": [
        "Impressionism emphasizes light and color over precise details.",
        "Digital art has revolutionized creative expression.",
        "Music theory explores harmony, rhythm, and melody.",
        "Street art can convey powerful social and political messages.",
        "Photography captures moments in time through light and perspective."
    ],
    "Space": [
        "Black holes have gravity so strong that not even light escapes.",
        "The Milky Way is a spiral galaxy containing billions of stars.",
        "NASA's Artemis program aims to return humans to the Moon.",
        "Mars rovers help scientists explore the Red Planet's surface.",
        "Space telescopes provide insights into distant galaxies."
    ],
    "Philosophy": [
        "Existentialism focuses on individual freedom and meaning.",
        "Ethics explores what is right and wrong in human behavior.",
        "Stoicism teaches emotional resilience and rationality.",
        "Logic studies valid reasoning and argumentation.",
        "The mind-body problem questions how consciousness arises."
    ],
    "Health": [
        "The immune system defends the body against infections.",
        "Mental health is crucial for overall well-being.",
        "Vaccines prevent the spread of infectious diseases.",
        "Exercise improves cardiovascular and cognitive health.",
        "Nutrition plays a vital role in maintaining energy and growth."
    ]
}

documents = []
for topic, sentences in topics.items():
    for sentence in sentences:
        documents.append(sentence)

#same id for same text
def doc_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
ids = [doc_id(doc) for doc in documents]

embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = [embedding_function.embed_query(text) for text in documents]


collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids
)

print(f"Ingested Successfully")
