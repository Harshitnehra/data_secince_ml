import os
from dotenv import load_dotenv
from langchain_mistralai.embeddings import MistralAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# Load env variables
load_dotenv()

# Initialize Mistral Embedding Model
embeddings = MistralAIEmbeddings(
    model="mistral-embed"
)

# Sample documents
documents = [
    "LangChain is a framework for building LLM applications.",
    "Mistral provides powerful embedding and chat models.",
    "Embeddings help in semantic search and similarity tasks.",
    "Cricket is a popular sport in India.",
    "Football is played worldwide."
]

# Step 1: Convert documents to embeddings
doc_vectors = embeddings.embed_documents(documents)

# Step 2: Input query
query = input("give me text : ", )

# Step 3: Convert query to embedding
query_vector = embeddings.embed_query(query)

# Step 4: Compute similarity
similarities = cosine_similarity([query_vector], doc_vectors)[0]

# Step 5: Get most similar document
most_similar_index = similarities.argmax()

print("Query:", query)
print("\nMost Similar Document:")
print(documents[most_similar_index])

print("\nSimilarity Scores:")
for i, score in enumerate(similarities):
    print(f"Doc {i}: {score:.4f}")

top_k = 3
top_indices = similarities.argsort()[-top_k:][::-1]

for i in top_indices:
    print(documents[i], similarities[i])