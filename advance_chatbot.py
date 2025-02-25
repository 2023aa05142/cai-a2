from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from basic_chatbot import BasicChatbot, ChatbotResponse

class AdvanceChatbot (BasicChatbot):
    def __init__(self, store, tokenizer, llm):
        self.store = store
        self.tokenizer = tokenizer
        self.llm = llm

    def __bm25_search(self, query: str, top_k: int = 5) -> List[str]:
        tokenized_chunks = [chunk.split() for chunk in self.store.all_chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.store.all_chunks[i] for i in top_indices]

    def __re_rank(self, query: str, initial_results: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[str]:
        #model = SentenceTransformer(model_name)
        query_embedding = self.store.model.encode([query], convert_to_numpy=True)
        result_embeddings = self.store.model.encode(initial_results, convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, result_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1]
        return [initial_results[i] for i in ranked_indices]

    def answer(self, query):
        bm25_results = self.__bm25_search(query)
        embedding_results = self._retrieve_similar_chunks(query)
        combined_results = bm25_results + embedding_results
        results = self.__re_rank(query, combined_results)
        response, confidence = self._generate_response(results, query)
        return ChatbotResponse(query=query, answer=response, confidence=confidence, chunks=results[:5])
