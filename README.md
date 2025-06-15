# Task Scheduler General Purpose Chatbot

A versatile chatbot designed to help you schedule tasks, manage reminders, and interact with various productivity tools using natural language. Powered by advanced language models, this project aims to streamline your workflow and boost productivity.

---

## Features

- **Task Scheduling:** Add, update, and remove tasks with simple commands.
- **Conversational Interface:** Interact naturally using chat.
- **Extensible:** Easily integrate with other productivity tools.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Task-Scheduler-General-purpose-Chatbot.git
cd Task-Scheduler-General-purpose-Chatbot
```

### 2. Install Dependencies

Make sure you have [Python 3.8+](https://www.python.org/downloads/) installed.

Install required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Download and Prepare the Language Model

#### Option 1: Using LM Studio

1. Download the [Gemma model](https://huggingface.co/google/gemma-2b) from Hugging Face.
2. Open LM Studio and import the model.
3. Export the model in `gguf` format.
4. Place the `.gguf` file in the `models/` directory of this repo or any other drive in your pc.

#### Option 2: Direct from Hugging Face

1. Install `huggingface_hub`:

    ```bash
    pip install huggingface_hub
    ```

2. Download the model:

    ```python
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="google/gemma-2b", filename="model.gguf", local_dir="models")
    ```

---

## Usage

Start the chatbot:

```bash
python main.py
```

Follow the on-screen instructions to interact with the chatbot.

---

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## License

This project is licensed under the MIT License.

---

**Happy Scheduling!**