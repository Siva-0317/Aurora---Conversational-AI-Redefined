import tkinter as tk
from tkinter import filedialog
from threading import Thread
from transformers import BertTokenizer, BertForSequenceClassification
from llama_cpp import Llama
import torch
from datetime import datetime
import time
import re

# === Load Models ===
intent_model = BertForSequenceClassification.from_pretrained("bert-intent-model")
intent_tokenizer = BertTokenizer.from_pretrained("bert-intent-model")

llm = Llama(
    model_path="D:/AI/lmstudio-community/gemma-2-2b-it-GGUF/gemma-2-2b-it-Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=6,
    use_mlock=True
)

# === Memory & State ===
user_memory = {"name": "yourname", "last_schedule": None} # Update to your username
conversation_history = []
stop_response = False

# === Intent Classifier ===
def classify_intent(text):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = intent_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted = torch.argmax(probs).item()
    return "schedule" if predicted == 1 else "chat"

# === Enhanced Prompt Generator ===
def generate_prompt(user_input, intent, user_memory, conversation_history):
    prompt_parts = [
        f"User's name: {user_memory.get('name', 'yourname')}",
        f"Current time: {datetime.now().strftime('%H:%M')}",
    ]
    
    # Only include relevant conversation history for schedule requests
    if intent == "schedule":
        prompt_parts.append("Current conversation context:")
        # Only include the most recent exchange for schedule requests
        if conversation_history:
            last_exchange = conversation_history[-1]
            prompt_parts.append(f"{last_exchange[0]}: {last_exchange[1]}")
    else:
        prompt_parts.append("Conversation history:")
        # Add conversation history (last 5 exchanges) for normal chat
        for speaker, message in conversation_history[-5:]:
            prompt_parts.append(f"{speaker}: {message}")

    if intent == "schedule":
        prompt_parts.extend([
            "\n[IMPORTANT SCHEDULE INSTRUCTIONS]",
            "1. Create ONE completely new schedule based ONLY on current request",
            "2. Format as markdown table with Time|Task|Notes columns",
            "3. Never repeat previous schedules or time blocks",
            "4. Include current time in schedule",
            "5. Ignore any previous conversation that isn't about scheduling",
            "6. If it involves studying or learning, make sure to allocate atleast 30mins time for it and include breaks",
            "7. Conclude in an artisitc and motivational tone",
            f"\nUser request: \"{user_input}\""
        ])
    else:
        prompt_parts.extend([
            "\n[CHAT INSTRUCTIONS] Respond conversationally as Aurora:",
            "1. Be creative and artistic",
            "2. Maintain natural flow from previous conversation",
            "3. Keep responses concise but meaningful"
        ])

    prompt_parts.append(f"\nUser: {user_input}")
    prompt_parts.append("Aurora:")
    return "\n".join(prompt_parts)

# === Improved Response Generator ===
def call_gemma_typing(prompt, callback):
    global stop_response
    stop_response = False
    
    output = llm(
        prompt,
        max_tokens=1024,
        stop=["</s>", "User:", "Assistant:", "Aurora:","yourname:","You:","**[Continue Conversation]**","```","Please continue the conversation"],
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.2
    )
    
    response = output["choices"][0]["text"].strip()
    
    # More aggressive cleaning for schedule responses
    if "[IMPORTANT SCHEDULE INSTRUCTIONS]" in prompt:
        # Remove any conversational parts that might have leaked in
        response = response.split("```")[0]  # Take only the first markdown block
        response = re.sub(r'(?i)(previous|earlier|before).*?(schedule|plan)', '', response)
    
    # Clean up response prefixes
    for prefix in ["Aurora:", "Response:"]:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    # Remove duplicate lines (more strict for schedules)
    seen = set()
    clean_lines = []
    for line in response.split('\n'):
        stripped = line.strip()
        if stripped not in seen and stripped:  # Also ignore empty lines
            clean_lines.append(line)
            seen.add(stripped)
    response = '\n'.join(clean_lines)
    
    # Typing effect
    typed = ""
    for char in response:
        if stop_response:
            break
        typed += char
        callback(typed)
        time.sleep(0.02)
    return typed

# === Enhanced GUI Application ===
class ArtMindApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 ArtMind Assistant - Aurora")
        self.root.geometry("1100x600")
        self.root.configure(bg="#fef6e4")

        # Left decoration
        self.left_art = tk.Label(root, bg="#fef6e4", text="🎨", font=("Segoe UI Emoji", 50))
        self.left_art.pack(side=tk.LEFT, padx=20)

        # Right decoration
        self.right_art = tk.Label(root, bg="#fef6e4", text="🖌️", font=("Segoe UI Emoji", 50))
        self.right_art.pack(side=tk.RIGHT, padx=20)

        # Main text area
        self.text_frame = tk.Frame(root, bg="#fef6e4")
        self.text_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.text_widget = tk.Text(
            self.text_frame, 
            wrap=tk.WORD, 
            font=("Consolas", 12), 
            bg="#fff8f0", 
            fg="#333",
            padx=10,
            pady=10
        )
        self.text_widget.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.text_widget.insert(tk.END, "👩‍🎨 Aurora: Hello <user>! How can I assist you today?\n\n") #update to your username
        self.text_widget.configure(state='disabled')

        self.scrollbar = tk.Scrollbar(self.text_frame, command=self.text_widget.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_widget['yscrollcommand'] = self.scrollbar.set

        # Input area
        self.entry = tk.Entry(
            root, 
            font=("Segoe UI", 12),
            relief=tk.FLAT,
            highlightthickness=2,
            highlightbackground="#a0c4ff",
            highlightcolor="#a0c4ff"
        )
        self.entry.pack(fill=tk.X, padx=20, pady=(0, 10))
        self.entry.bind("<Return>", lambda e: self.send_input())
        self.entry.focus_set()

        # Button frame
        self.button_frame = tk.Frame(root, bg="#fef6e4")
        self.button_frame.pack(pady=(0, 10))

        # Buttons
        self.send_button = tk.Button(
            self.button_frame, 
            text="Send", 
            command=self.send_input, 
            bg="#a0c4ff",
            activebackground="#8ab4ff",
            fg="white",
            relief=tk.FLAT,
            padx=15
        )
        self.send_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(
            self.button_frame, 
            text="Stop", 
            command=self.stop_generation, 
            bg="#ffadad",
            activebackground="#ff9b9b",
            fg="white",
            relief=tk.FLAT,
            padx=15
        )
        self.stop_button.pack(side=tk.LEFT)

        # Clear context button
        self.clear_button = tk.Button(
            self.button_frame, 
            text="Clear Context", 
            command=self.clear_context, 
            bg="#caffbf",
            activebackground="#b0ffa3",
            fg="white",
            relief=tk.FLAT,
            padx=15
        )
        self.clear_button.pack(side=tk.LEFT, padx=10)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(
            root, 
            textvariable=self.status_var,
            bg="#fef6e4",
            fg="#666",
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, padx=20, pady=(0, 5))

    def update_text(self, text, is_user=False):
        self.text_widget.configure(state='normal')
        prefix = "You: " if is_user else "Aurora: "
        self.text_widget.insert(tk.END, "\n" + prefix + text + "\n")
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def stop_generation(self):
        global stop_response
        stop_response = True
        self.status_var.set("Generation stopped")

    def clear_context(self):
        global conversation_history
        conversation_history = []
        self.text_widget.configure(state='normal')
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.insert(tk.END, "👩‍🎨 Aurora: Conversation history cleared. Hello again <user>!\n\n") #update to your username
        self.text_widget.configure(state='disabled')
        self.status_var.set("Conversation context cleared")

    def send_input(self):
        user_input = self.entry.get().strip()
        if not user_input:
            return

        self.entry.delete(0, tk.END)
        self.update_text(user_input, is_user=True)
        conversation_history.append(("You", user_input))

        # Classify intent
        intent = classify_intent(user_input)
        self.status_var.set(f"Detected intent: {intent}")
        
        # Generate prompt
        prompt = generate_prompt(user_input, intent, user_memory, conversation_history)

        def generate():
            try:
                full_response = ""
                
                def display_callback(txt):
                    nonlocal full_response
                    full_response = txt
                    self.text_widget.configure(state='normal')
                    # Find the last Aurora response
                    last_aurora_pos = self.text_widget.search("Aurora:", "end-1c", backwards=True)
                    if last_aurora_pos:
                        self.text_widget.delete(last_aurora_pos, "end")
                    self.text_widget.insert(tk.END, "Aurora: " + full_response + "\n")
                    self.text_widget.see(tk.END)
                    self.text_widget.configure(state='disabled')
                    self.root.update()

                self.status_var.set("Generating response...")
                call_gemma_typing(prompt, display_callback)
                
                # Clean up any remaining table formatting issues
                full_response = re.sub(r'\|\s+\|', '|', full_response)
                
                conversation_history.append(("Aurora", full_response))
                self.status_var.set("Ready")
                
                # Update memory if schedule
                if intent == "schedule":
                    user_memory["last_schedule"] = {
                        "time": datetime.now(),
                        "content": full_response
                    }
                    
            except Exception as e:
                self.status_var.set(f"Error: {str(e)}")
                self.update_text(f"Sorry, I encountered an error: {str(e)}")

        Thread(target=generate).start()

# === Start App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = ArtMindApp(root)
    root.mainloop()