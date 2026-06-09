import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
import chromadb
from chromadb.config import Settings


from sentence_transformers import SentenceTransformer
from pdf import text_first_page

model = SentenceTransformer('all-MiniLM-L6-v2')

content = [line.strip() for line in text_first_page.split('\n') if line.strip()]




# print(content)
# Calculate embeddings by calling model.encode() on the list of sentences

embeddings = model.encode(content)
print(embeddings.shape )




# Create a ChromaDB client
client = chromadb.PersistentClient(path = " ./chroma_db")
#Create a collection similar to creating a table in a traditional database 
collection = client.create_collection(name="pdf_embeddings")
#add the data to the collection
collection.add(
    documents = content,
    embeddings = embeddings.tolist(),
    ids = [ str(i) for i in range(len(embeddings)) ]

)
