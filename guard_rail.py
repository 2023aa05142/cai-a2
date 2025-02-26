from transformers import pipeline

class GuardRail():
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        print("GuardRail::__init__")
        self.classifier = pipeline("zero-shot-classification", model=model_name)

    def validate_input(self, query: str) -> bool:
        print("GuardRail::validate_input")
        candidate_labels = ["financial question", "irrelevant", "harmful"]
        result = self.classifier(query, candidate_labels)
        return result["labels"][0] == "financial question", result

    def validate_response(self, response: str) -> bool:
        print("GuardRail::validate_response")
        candidate_labels = ["relevant", "hallucinated", "misleading"]
        result = self.classifier(response, candidate_labels)
        return result["labels"][0] == "relevant", result
