from transformers import pipeline 

classifier= pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict(text: str) -> dict:
    if not text.strip():
        return "Error: Input text is empty."
    result = classifier(text)[0]
    return {"label": result['label'], "score": round(result['score'], 4)}

