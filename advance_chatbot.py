from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

from basic_chatbot import BasicChatbot, ChatbotResponse

class AdvanceChatbot (BasicChatbot):
    def __init__(self, store, tokenizer, llm):
        print("AdvanceChatbot::__init__")
        self.store = store
        self.tokenizer = tokenizer
        self.llm = llm

    def __bm25_search(self, query: str, top_k: int = 5) -> List[str]:
        print("AdvanceChatbot::__bm25_search")
        tokenized_chunks = [chunk.split() for chunk in self.store.all_chunks]
        bm25 = BM25Okapi(tokenized_chunks)
        scores = bm25.get_scores(query.split())
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.store.all_chunks[i] for i in top_indices]

    def __re_rank(self, query: str, initial_results: List[str]) -> List[str]:
        print("AdvanceChatbot::__re_rank")
        query_embedding = self.store.model.encode([query], convert_to_numpy=True)
        result_embeddings = self.store.model.encode(initial_results, convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, result_embeddings)[0]
        ranked_indices = np.argsort(similarities)[::-1]
        return [initial_results[i] for i in ranked_indices]

    def __multi_stage_retrieval(self, query: str) -> List[str]:
        print("AdvanceChatbot::__multi_stage_retrieval")
        # Stage 1: BM25 retrieval
        bm25_results = self.__bm25_search(query)
        # Stage 2: Embedding-based retrieval
        embedding_results = self._retrieve_similar_chunks(query)
        # Combine and deduplicate results
        combined_results = bm25_results + embedding_results
        # Stage 3: Re-rank results
        return self.__re_rank(query, combined_results)

    def __filter_relevant_retrievals(self, responses: List[str], query: str, threshold) -> List[str]:
        print("AdvanceChatbot::__filter_relevant_retrievals")
        query_embedding = self.store.model.encode([query], convert_to_numpy=True)
        response_embeddings = self.store.model.encode(responses, convert_to_numpy=True)
        similarities = cosine_similarity(query_embedding, response_embeddings)[0]
        filtered_responses = [response for i, response in enumerate(responses) if similarities[i] >= threshold]
        return filtered_responses

    def answer(self, query, threshold = 0.5):
        print("AdvanceChatbot::answer")
        # get results using multi stage retrieval
        results = self.__multi_stage_retrieval(query)
        relevant_results = self.__filter_relevant_retrievals(results, query, threshold)
        print(relevant_results)
        # generate response
        response, confidence = self._generate_response(relevant_results, query)
        return ChatbotResponse(query=query, answer=response, confidence=confidence, chunks=results[:5])
