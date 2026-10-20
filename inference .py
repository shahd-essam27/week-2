from transformers import pipeline 

classifier= pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict(text: str) -> dict:
    if not text.strip():
        return "Error: Input text is empty."
    result = classifier(text)[0]
    return {"label": result['label'], "score": round(result['score'], 4)}

if __name__ == "__main__":
    test_text = "Building AI projects step-by-step with clean code is so satisfying!"
    
    print("\n--- Testing Pipeline Inference Module ---")
    result = predict(test_text)
    print(f"Input Text: {test_text}")
    print(f"Prediction: {result['label']} (Confidence: {result['score']})")
