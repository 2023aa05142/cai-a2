import torch
from typing import List

class ChatbotResponse:
    def __init__(self, query, answer, confidence, chunks):
        self.query = query
        self.answer = answer
        self.confidence = confidence
        self.chunks = chunks

class BasicChatbot:
    def __init__(self, store, tokenizer, llm):
        print("BasicChatbot::__init__")
        self.store = store
        self.tokenizer = tokenizer
        self.llm = llm
    
    def _retrieve_similar_chunks(self, query: str, top_k: int = 5) -> List[str]:
        print("BasicChatbot::_retrieve_similar_chunks")
        query_embedding = self.store.model.encode([query], convert_to_numpy=True)
        distances, indices = self.store.vector_db.search(query_embedding, top_k)
        return [self.store.all_chunks[i] for i in indices[0]]

    def _generate_response(self, chunks: List[str], query: str, max_new_tokens: int = 1024) -> str:
        print("BasicChatbot::_generate_response")
        context = "\n".join(chunks[:3])  # Use the top 3 chunks for context
        input_text = f"Query: {query}\nContext: {context}\nResponse:"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.llm.generate(
            inputs["input_ids"],
            return_dict_in_generate=True,
            output_scores=True,
            #max_length=max_new_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id  # Ensure proper padding
        )

        # Decode the generated response
        generated_ids = outputs.sequences[0]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Calculate confidence score as the average token probability
        scores = outputs.scores  # List of logits for each token
        probs = [torch.softmax(score, dim=-1) for score in scores]
        token_probs = [prob[0, token].item() for prob, token in zip(probs, generated_ids[1:])]  # Exclude input tokens
        confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

        return response, confidence

    def answer(self, query, threshold = 0.5):
        print("BasicChatbot::answer")
        results = self._retrieve_similar_chunks(query)
        response, confidence = self._generate_response(results, query)
        return ChatbotResponse(query=query, answer=response, confidence=confidence, chunks=results[:5])
