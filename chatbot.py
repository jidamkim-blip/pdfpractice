

from sentence_transformers import SentenceTransformer
from embedding import collection
import anthropic



#create question and encode it to get the embedding vector for the question
model = SentenceTransformer('all-MiniLM-L6-v2')
question = ["which professional experience is most relevant for Data Science Jobs?"]

question_embedding = model.encode(question)

#query ChromaDB to get most similar embedding to the question 
results = collection.query(
    query_embeddings = question_embedding.tolist(),
    n_results = 2
)

documents = results.get('documents') if isinstance(results, dict) else None
if documents and documents[0]:
    context = "\n".join(documents[0])
else:
    context = ""

client = anthropic.Anthropic()
message = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1000,
    system=f"You are a helpful assistant that answers questions based on the provided context.\n\nContext:\n{context}",
    messages=[
        {"role": "user", "content": question[0]}
    ]
)

print(message.content[0].text)
