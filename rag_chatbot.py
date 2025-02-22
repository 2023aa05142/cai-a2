class ragChatbotResponse:
    def __init__(self, query, answer, score):
        self.query = query
        self.answer = answer
        self.score = score

class ragChatbot:
    def __init__(self):
        pass
    def answer(self, query):
        return ragChatbotResponse(query=query, answer='Not implemented', score= 0.0)
