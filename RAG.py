import torch
import torch.nn as nn

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class RAG_module():
    def __init__(self, chunk_size=100, chunk_overlap=20, length_function=len):
        self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = chunk_size,
                    chunk_overlap = chunk_overlap,
                    length_function = length_function)
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.embedding_dicts = []
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def initalize_embedding_database(self, text):
        # Split token into chunks
        text = self.text_splitter.create_documents(text)

        for chunk in text:
            content = chunk.page_content
            embedding = self.model.encode(content, convert_to_tensor=True)
            embedding_dict = {"sentence": content, "embedding": embedding.tolist()}
            self.embedding_dicts.append(embedding_dict)
    
    def find_top_k_embedding(self, query, k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = []
        for chunk in self.embedding_dicts:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            score = self.cos(torch.tensor(chunk["embedding"]).unsqueeze(0).to(device), query_embedding.unsqueeze(0).to(device))
            scores.append(score)

        top_k_indices = torch.topk(input=torch.tensor(scores), k=k).indices.tolist()
        top_k_chunks = [self.embedding_dicts[i.item()]["sentence"] for i in top_k_indices]
        return top_k_chunks

