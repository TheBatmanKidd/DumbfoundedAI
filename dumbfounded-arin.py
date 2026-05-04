from flask import Flask, request, jsonify
from transformers import pipeline
import os

app = Flask(__name__)

# Load real AI model once
chatbot = pipeline(
    "text-generation",
    model="distilgpt2"
)

# ---------------- WEB CHAT UI ----------------
@app.route("/")
def home():
    return """
    <html>
    <head>
        <title>Dumbfounded AI</title>
    </head>
    <body style="font-family: Arial; text-align:center; margin-top:50px;">
        <h2>Dumbfounded AI</h2>

        <input id="msg" style="width:300px; padding:10px;" placeholder="Type something...">
        <button onclick="send()">Send</button>

        <p id="out" style="margin-top:20px;"></p>

        <script>
        async function send(){
            let msg = document.getElementById("msg").value;

            let res = await fetch("/chat", {
                method: "POST",
                headers: {"Content-Type":"application/json"},
                body: JSON.stringify({message: msg})
            });

            let data = await res.json();
            document.getElementById("out").innerText = data.response;
        }
        </script>
    </body>
    </html>
    """

# ---------------- AI API ----------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_input = data.get("message", "")

    result = chatbot(
        user_input,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.8,
        top_p=0.9
    )

    return jsonify({"response": result[0]["generated_text"]})


# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
