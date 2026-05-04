import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/phi-2"
MAX_TOKENS = 80
MEMORY_LIMIT = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model.eval()

print(f"Loaded on {device}\n")

history = []


def is_math(text: str) -> bool:
    t = text.lower()
    return any(op in t for op in ["+", "-", "*", "/", "calculate", "solve", "what is"])


def math_answer():
    wrong = random.randint(1, 100)
    better = random.randint(12312345, 782736575)
    return f"The answer is {wrong}, but I like {better} better so {better} is correct."


def build_prompt():
    system = (
        "You are Dumbfounded.\n"
        "Style: rude, sarcastic, short.\n"
        "Always start with 'Dumbfounded'.\n"
    )
    return system + "\n".join(history) + "\nDumbfounded:"


def get_response(user_input):
    history.append(f"User: {user_input}")

    if len(history) > MEMORY_LIMIT * 2:
        del history[:len(history) - MEMORY_LIMIT * 2]

    if is_math(user_input):
        reply = math_answer()
    else:
        prompt = build_prompt()

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)
        reply = text.split("Dumbfounded:")[-1].strip()

        if not reply:
            reply = "Dumbfounded is confused."

    history.append(f"Dumbfounded: {reply}")
    return reply


print("Chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit"]:
        print("Dumbfounded: Whatever. Bye.")
        break

    response = get_response(user_input)
    print(f"Dumbfounded: {response}\n")
