import time
from transformers import pipeline

print("Loading models for comparative benchmarking...")

models_to_test = {
    "DistilBERT (Fast)": pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english"),
    "RoBERTa (Robust)": pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
}

# Test samples along with their "ground truth" (expected sentiment for quality check)
test_samples = [
    {"text": "This product completely exceeded my expectations, absolutely love it!", "expected": "POSITIVE"},
    {"text": "Worst customer service I have ever experienced. Total waste of money.", "expected": "NEGATIVE"},
    {"text": "The build quality is decent, but nothing special for the price.", "expected": "NEUTRAL"}, # Note: DistilBERT maps NEUTRAL or defaults, let's see how they handle it!
    {"text": "An absolute masterpiece of engineering and software design!", "expected": "POSITIVE"},
    {"text": "It broke down on the very first day of use. Terrible quality.", "expected": "NEGATIVE"}
]

def benchmark_models(models_dict, samples):
    print(f"\n--- Starting Comparative Benchmark & Quality Evaluation ---")
    
    for model_name, classifier in models_dict.items(): # return key ,value 
        print(f"\nEvaluating: {model_name}...")
        latencies = []
        correct_predictions = 0
        total_confidence = 0
        
        for sample in samples:
            text = sample["text"]
            expected = sample["expected"]
            
            start_time = time.time()
            result = classifier(text)[0]
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            pred_label = result['label'].upper()
            confidence = result['score']
            total_confidence += confidence
            
            # Simple accuracy check (handling label text variations)
            if expected in pred_label or pred_label in expected:
                correct_predictions += 1
                
        avg_latency = sum(latencies) / len(latencies)
        avg_confidence = total_confidence / len(samples)
        accuracy = (correct_predictions / len(samples)) * 100
        
        print(f" -> Average Latency: {avg_latency:.2f} ms")
        print(f" -> Average Confidence (Quality): {avg_confidence:.4f}")
        print(f" -> Accuracy on test suite: {accuracy:.0f}%")

if __name__ == "__main__":
    benchmark_models(models_to_test, test_samples)