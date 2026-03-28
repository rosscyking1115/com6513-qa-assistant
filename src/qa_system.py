from transformers import pipeline

class QAWrapper:
    def __init__(self, model_name):
        self.qa_pipeline = pipeline("question-answering", model=model_name)

    def answer_question(self, context, question):
        return self.qa_pipeline(question=question, context=context)