import torch
import torch.nn as nn
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

class RAG_module():
    def __init__(self, chunk_size=500, chunk_overlap=20, length_function=len, embedding_dicts=[]):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function)
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.embedding_dicts = embedding_dicts
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def initialize_embedding_database(self, text):
        text_chunks = self.text_splitter.create_documents(text)
        for chunk in text_chunks:
            content = chunk.page_content
            embedding = self.model.encode(content, convert_to_tensor=True)
            embedding_dict = {"sentence": content, "embedding": embedding.cpu().numpy()}
            self.embedding_dicts.append(embedding_dict)

    def find_top_k_embedding(self, query, k=20):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        query_embedding = self.model.encode(query, convert_to_tensor=True).to(device)
        scores = []
        for chunk in self.embedding_dicts:
            chunk_embedding = torch.tensor(chunk["embedding"], device=device)
            score = self.cos(chunk_embedding.unsqueeze(0), query_embedding.unsqueeze(0))
            scores.append(score.item())
        k = min(k, len(scores))
        top_k_indices = torch.topk(torch.tensor(scores), k=k).indices.tolist()
        top_k_chunks = [self.embedding_dicts[i]["sentence"] for i in top_k_indices]
        return top_k_chunks
