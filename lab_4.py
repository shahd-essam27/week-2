import gradio as gr
from inference import predict

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