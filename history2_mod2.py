from transformers import BertTokenizer, BertForSequenceClassification
from llama_cpp import Llama
import torch
from datetime import datetime

# === Load Fine-Tuned BERT Model ===
intent_model = BertForSequenceClassification.from_pretrained("bert-intent-model")
intent_tokenizer = BertTokenizer.from_pretrained("bert-intent-model")

# === Load GGUF Gemma Model ===
llm = Llama(
    model_path="D:/AI/lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    use_mlock=True
)

# === Conversation History ===
conversation_history = []  # Stores tuples of (speaker, message)

# === User Memory ===
user_memory = {
    "name": "Risha",
    "last_schedule": None
}

# === Intent Classifier ===
def classify_intent(text):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = intent_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted = torch.argmax(probs).item()
    return "schedule" if predicted == 1 else "chat"

# === Context-Aware Prompt Builder ===
def generate_prompt(user_input, intent, user_memory, conversation_history):
    # Base context with memory
    prompt_parts = [
        f"User's name: {user_memory.get('name', 'Risha')}",
        f"Current time: {datetime.now().strftime('%H:%M')}",
        "Conversation history:"
    ]
    
    # Add conversation history (last 4 exchanges max)
    for speaker, message in conversation_history[-8:]:
        prompt_parts.append(f"{speaker}: {message}")
    
    # Intent-specific instructions
    if intent == "schedule":
        prompt_parts.extend([
            "\n[INSTRUCTIONS] Create a detailed schedule with time blocks.",
            f"User request: \"{user_input}\"",
            "Format as a clean table with Time|Task|Notes columns."
        ])
    else:
        prompt_parts.extend([
            "\n[INSTRUCTIONS] Respond conversationally without prefixes.",
            "Answer directly when asked about known information."
        ])
    
    prompt_parts.append(f"\nUser: {user_input}")
    prompt_parts.append("Response:")  # Changed from "Assistant:"
    return "\n".join(prompt_parts)

# === LLM Call ===
def call_gemma_locally(prompt):
    output = llm(
        prompt,
        max_tokens=512,
        stop=["</s>", "User:", "Assistant:"],  # Added stop tokens
        temperature=0.7,
        top_p=0.9
    )
    # Clean up response
    response = output["choices"][0]["text"].strip()
    # Remove any remaining prefix patterns
    for prefix in ["Assistant:", "Response:"]:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    return response

# === Update Memory ===
def update_memory(intent, user_input, response):
    if intent == "schedule":
        user_memory["last_schedule"] = {
            "time": datetime.now(),
            "content": response
        }

# === Main Loop ===
print("\n🌟 Assistant ready! Type 'exit' to quit.\n")
while True:
    try:
        user_input = input("\n💬 You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break

        # Classify intent and update history
        intent = classify_intent(user_input)
        print(f"🧠 Detected Intent: {intent}")
        conversation_history.append(("User", user_input))

        # Generate and process response
        prompt = generate_prompt(user_input, intent, user_memory, conversation_history)
        print(f"📤 Prompting Gemma...")
        
        response = call_gemma_locally(prompt)
        print(f"\n🤖 Gemma: {response}")
        
        # Update systems
        conversation_history.append(("Assistant", response))
        update_memory(intent, user_input, response)
        
        # Trim history if too long
        if len(conversation_history) > 12:
            conversation_history = conversation_history[-12:]

    except KeyboardInterrupt:
        print("\n🛑 Session ended by user")
        break
    except Exception as e:
        print(f"\n⚠️ Error: {str(e)}")
        continue