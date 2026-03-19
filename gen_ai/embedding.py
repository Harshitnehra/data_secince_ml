import os
from dotenv import load_dotenv
from langchain_mistralai.embeddings import MistralAIEmbeddings

# Load environment variables
load_dotenv()

# Initialize embedding model
embeddings = MistralAIEmbeddings(
    model="mistral-embed" 
)

# Example text
text = "LangChain makes working with LLMs easy."

# Generate embedding
embedding_vector = embeddings.embed_query(text)

# Print result
print("Embedding vector length:", len(embedding_vector))
print("First 10 values:", embedding_vector)