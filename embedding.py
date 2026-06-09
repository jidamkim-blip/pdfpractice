import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''


from sentence_transformers import SentenceTransformer
from pdf import text_first_page

model = SentenceTransformer('all-MiniLM-L6-v2')

content = text_first_page.split('\n')

# print(content)
# Calculate embeddings by calling model.encode() on the list of sentences

embeddings = model.encode(content)
print(embeddings.shape )




