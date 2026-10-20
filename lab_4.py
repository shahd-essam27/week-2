import gradio as gr
 #from lab2_2 import predict
from transformers import pipeline

# Initialize the classifier once when the file is imported
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
def predict(text: str) -> dict:
    if not text.strip():
        return {"error": "Input text is empty."} # Changed to a dict for return type consistency
    
    result = classifier(text)[0]
    return {"label": result['label'], "score": round(result['score'], 4)}

def classify_text(text):
    if not text.strip():
        return "Please enter some text to classify."
    
    # Call the modular inference function we built earlier
    result = predict(text)
    
    # Return formatted results
    return f"Label: {result['label']} | Confidence: {result['score']}"

# Build the Gradio web interface
demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(lines=3, placeholder="Type a sentence, review, or thought here..."),
    outputs="text",
    title="AI Sentiment Analysis Dashboard",
    description="Week 2 AI Engineering Lab - Powered by DistilBERT and Gradio."
)

if __name__ == "__main__":
    print("Launching Gradio web app...")
    demo.launch()