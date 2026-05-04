from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Load model ONLY once (important for memory)
chatbot = pipeline(
    "text-generation",
    model="distilgpt2"
)

@app.route("/")
def home():
    return "Dumbfounded AI is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")

    result = chatbot(
        user_input,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

    response = result[0]["generated_text"]

    return jsonify({"response": response})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
