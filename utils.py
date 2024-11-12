import json
from datetime import datetime
from langchain.schema.messages import HumanMessage, AIMessage

def save_chat_history_json(chat_history, file_path):
    # Saving chat history to JSON, assuming messages are dictionaries
    json_data = [message if isinstance(message, dict) else message.dict() for message in chat_history]
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4)  # Added indentation for readability

def load_chat_history_json(file_path):
    # Load JSON data and convert back to HumanMessage or AIMessage objects
    with open(file_path, "r") as f:
        json_data = json.load(f)
        messages = [
            HumanMessage(**message) if message["type"] == "human" else AIMessage(**message)
            for message in json_data
        ]
        return messages

def get_timestamp():
    # Safe timestamp format for filenames
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
