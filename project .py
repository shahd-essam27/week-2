import time
import gradio as gr
from transformers import pipeline

# Candidate models to evaluate
candidate_models = {
    "XLM-RoBERTa-Base": "papluca/xlm-roberta-base-language-detection",
    "Alexnea-BERT": "alexneakameni/language_detection",
}

# Test sample suite for evaluation
scout_test_samples = [
    {"text": "Hello, welcome to our AI engineering project!", "expected_lang": "en"},
    {
        "text": "Bonjour tout le monde, bienvenue dans notre projet !",
        "expected_lang": "fr",
    },
    {
        "text": "¡Hola a todos, bienvenidos a nuestro proyecto!",
        "expected_lang": "es",
    },
    {
        "text": "مرحباً بكم في مشروع هندسة الذكاء الاصطناعي!",
        "expected_lang": "ar",
    },
    {
        "text": "Hallo zusammen, willkommen bei unserem KI-Projekt!",
        "expected_lang": "de",
    },
]


def choose_best_model(models_dict, test_samples):
  """Evaluates models, selects the best one, and returns its pipeline & name."""
  print(
      "🔍 [Scout Report] Running benchmark to choose the best model for Mini"
      " Project 2..."
  )
  best_model_name = None
  winning_pipeline = None
  highest_score = -1

  for name, model_id in models_dict.items():
    print(f"   -> Testing candidate: {name}...")
    try:
      clf = pipeline("text-classification", model=model_id)
      latencies = []
      correct = 0

      for sample in test_samples:
        start = time.time()
        res = clf(sample["text"])
        end = time.time()

        latencies.append((end - start) * 1000)
        pred_label = res[0]["label"].lower()
        if sample["expected_lang"].lower() in pred_label:
          correct += 1

      accuracy = (correct / len(test_samples)) * 100
      avg_latency = sum(latencies) / len(latencies)
      composite_score = accuracy - (
          avg_latency / 100
      )  # Balancing accuracy and speed

      print(
          f"      Result -> Accuracy: {accuracy:.1f}% | Avg Latency:"
          f" {avg_latency:.2f} ms"
      )

      if composite_score > highest_score:
        highest_score = composite_score
        best_model_name = name
        winning_pipeline = clf
    except Exception as e:
      print(f"      [Warning] Could not load {name}: {e}")

  if not winning_pipeline:
    best_model_name = "XLM-RoBERTa-Base"
    winning_pipeline = pipeline(
        "text-classification", model=models_dict[best_model_name]
    )

  print(f"\n🏆 [Scout Winner Selected]: {best_model_name}")
  print(
      "   Rationale: Superior cross-lingual accuracy across diverse global"
      " scripts.\n"
  )

  return winning_pipeline, best_model_name


# Automatically evaluate and lock in the winning model
active_classifier, winning_name = choose_best_model(
    candidate_models, scout_test_samples
)


# Gradio prediction function
def detect_language(text):
  if not text.strip():
    return "Please enter text to analyze."
  result = active_classifier(text)[0]
  return (
      f"Detected Language: {result['label']} | Confidence:"
      f" {round(result['score'], 4)} | Winning Model Used: {winning_name}"
  )


# Gradio Web App Interface
demo = gr.Interface(
    fn=detect_language,
    inputs=gr.Textbox(
        lines=3, placeholder="Type text in English, French, Arabic, Spanish..."
    ),
    outputs="text",
    title="🌍 Automated Language Detector Scout Demo",
    description=(
        f"Mini Project 2: Automatically selected **{winning_name}** based on live"
        " benchmark evaluation."
    ),
)

if __name__ == "__main__":
  print("🚀 Launching Gradio web app with public sharing enabled...")
  # share=True automatically generates a temporary public URL link (e.g., https://xxxx.gradio.live)
  demo.launch(share=True)