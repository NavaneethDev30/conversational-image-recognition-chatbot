import streamlit as st
import os
import yaml
from PIL import Image
import logging
import io
from image_handler import handle_image
from llm_chains import load_normal_chain
from utils import save_chat_history_json, get_timestamp, load_chat_history_json
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from transformers import BlipProcessor, BlipForConditionalGeneration



# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Load configuration from YAML file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set environment variable for Hugging Face token (if needed)
os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_access_token_here"

# Load BLIP model and processor for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def load_chain(chat_history):
    """Load the LLM chain with provided chat history."""
    return load_normal_chain(chat_history)

def clear_input_field():
    """Clear the input field after submitting a question."""
    st.session_state.user_question = st.session_state.user_input
    st.session_state.user_input = ""

def set_send_input():
    """Set send input flag and clear input field."""
    st.session_state.send_input = True
    clear_input_field()

def save_chat_history():
    """Save the chat history as a JSON file."""
    if "history" not in st.session_state or not st.session_state.history:
        return  # Nothing to save if history is empty

    session_key = st.session_state.session_key
    file_path = os.path.join(config["chat_history_path"], f"{session_key}.json")
    save_chat_history_json(st.session_state.history, file_path)

def handle_image_description(image_bytes):
    """Process the uploaded image and generate a description using BLIP model."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        description = processor.decode(out[0], skip_special_tokens=True)
        logging.info(f"Generated image description: {description}")
        return description
    except Exception as e:
        logging.error(f"Error processing the image: {e}")
        return "An error occurred while processing the image."

def add_to_chat_history(user_question, answer):
    """Add unique user question and AI response to chat history."""
    # Check if the question-answer pair already exists to avoid duplicates
    if not st.session_state.history or st.session_state.history[-2:] != [HumanMessage(content=user_question), AIMessage(content=answer)]:
        st.session_state.history.append(HumanMessage(content=user_question))
        st.session_state.history.append(AIMessage(content=answer))

def main():
    st.title('Multimodal Local Chat App')
    chat_container = st.container()
    st.sidebar.title("Chat Session")

    if "send_input" not in st.session_state:
        st.session_state.send_input = False
        st.session_state.user_question = ""

    if "session_key" not in st.session_state:
        st.session_state.session_key = get_timestamp()  # Generate new session key

    # Load previous chat sessions
    chat_sessions = ["new_session"] + [
        file.replace(".json", "") for file in os.listdir(config["chat_history_path"]) if file.endswith(".json")
    ]
    selected_session = st.sidebar.selectbox("Select a chat session", chat_sessions, index=0)

    # Load chat history based on session selection
    if selected_session == "new_session":
        st.session_state.history = []  # Clear chat history for a fresh session
        st.session_state.session_key = get_timestamp()  # Generate new session key
    else:
        file_path = os.path.join(config["chat_history_path"], f"{selected_session}.json")
        st.session_state.history = load_chat_history_json(file_path)
        st.session_state.session_key = selected_session  # Set session key to the selected session

    # Initialize the chat chain and history
    chat_history = StreamlitChatMessageHistory(key="history")
    llm_chain = load_chain(chat_history)

    # Sidebar file uploader for image
    uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    user_input = st.text_input('Type your message here', key='user_input', on_change=set_send_input)
    send_button = st.button('Send', key='send_button')

    if uploaded_image:
        with st.spinner("Processing image..."):
            description = handle_image_description(uploaded_image.getvalue())
            st.session_state.image_description = description
            add_to_chat_history("Please describe this image.", description)

    if send_button or st.session_state.send_input:
        if st.session_state.user_question:
            # Determine answer based on available image description
            if "image_description" in st.session_state and (
                'image' in st.session_state.user_question.lower() or 'describe' in st.session_state.user_question.lower()
            ):
                answer = st.session_state.image_description
            else:
                answer = llm_chain.run(st.session_state.user_question)

            # Add question and answer to chat history
            add_to_chat_history(st.session_state.user_question, answer)

            # Display user input and AI response in the container
            with chat_container:
                st.chat_message('user').write(st.session_state.user_question)
                st.chat_message('ai').write(answer)
            
            # Reset input flags
            st.session_state.user_question = ""
            st.session_state.send_input = False

    # Display the entire chat history from session state
    if st.session_state.history:
        with chat_container:
            st.write("Chat History:")
            for message in st.session_state.history:
                if isinstance(message, HumanMessage):
                    st.chat_message("user").write(message.content)
                elif isinstance(message, AIMessage):
                    st.chat_message("ai").write(message.content)

    save_chat_history()

if __name__ == '__main__':
    main()






















































































































































# import streamlit as st
# import os
# import yaml
# from PIL import Image
# import logging
# import io
# from image_handler import handle_image
# from llm_chains import load_normal_chain
# from utils import save_chat_history_json, get_timestamp, load_chat_history_json
# from langchain_community.chat_message_histories import StreamlitChatMessageHistory
# from langchain_core.messages import HumanMessage, AIMessage
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # Set up logging for debugging
# logging.basicConfig(level=logging.INFO)

# # Load configuration from YAML file
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# # Set environment variable for Hugging Face token (if needed)
# os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_access_token_here"

# # Load BLIP model and processor for image captioning
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# def load_chain(chat_history):
#     """Load the LLM chain with provided chat history."""
#     return load_normal_chain(chat_history)

# def clear_input_field():
#     """Clear the input field after submitting a question."""
#     st.session_state.user_question = st.session_state.user_input
#     st.session_state.user_input = ""

# def set_send_input():
#     """Set send input flag and clear input field."""
#     st.session_state.send_input = True
#     clear_input_field()

# def save_chat_history():
#     """Save the chat history as a JSON file."""
#     if "history" not in st.session_state or not st.session_state.history:
#         return  # Nothing to save if history is empty

#     session_key = st.session_state.session_key
#     file_path = os.path.join(config["chat_history_path"], f"{session_key}.json")
#     save_chat_history_json(st.session_state.history, file_path)

# def handle_image_description(image_bytes):
#     """Process the uploaded image and generate a description using BLIP model."""
#     try:
#         image = Image.open(io.BytesIO(image_bytes))
#         inputs = processor(image, return_tensors="pt")
#         out = model.generate(**inputs)
#         description = processor.decode(out[0], skip_special_tokens=True)
#         logging.info(f"Generated image description: {description}")
#         return description
#     except Exception as e:
#         logging.error(f"Error processing the image: {e}")
#         return "An error occurred while processing the image."

# def add_to_chat_history(user_question, answer):
#     """Add user question and AI response to chat history, ensuring no duplicates."""
#     # Check if last AI message in history matches the answer, to avoid duplication
#     if not st.session_state.history or (st.session_state.history[-1].content != answer):
#         st.session_state.history.append(HumanMessage(content=user_question))
#         st.session_state.history.append(AIMessage(content=answer))

# def main():
#     st.title('Multimodal Local Chat App')
#     chat_container = st.container()
#     st.sidebar.title("Chat Session")

#     if "send_input" not in st.session_state:
#         st.session_state.send_input = False
#         st.session_state.user_question = ""

#     if "session_key" not in st.session_state:
#         st.session_state.session_key = get_timestamp()  # Generate new session key

#     chat_sessions = ["new_session"] + [
#         file.replace(".json", "") for file in os.listdir(config["chat_history_path"]) if file.endswith(".json")
#     ]
#     selected_session = st.sidebar.selectbox("Select a chat session", chat_sessions, index=0)

#     if selected_session == "new_session":
#         st.session_state.history = []  # Clear chat history for a fresh session
#         st.session_state.session_key = get_timestamp()  # Generate new session key
#     else:
#         file_path = os.path.join(config["chat_history_path"], f"{selected_session}.json")
#         st.session_state.history = load_chat_history_json(file_path)
#         st.session_state.session_key = selected_session  # Set session key to the selected session

#     chat_history = StreamlitChatMessageHistory(key="history")
#     llm_chain = load_chain(chat_history)

#     uploaded_image = st.sidebar.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
#     user_input = st.text_input('Type your message here', key='user_input', on_change=set_send_input)
#     send_button = st.button('Send', key='send_button')

#     if uploaded_image:
#         with st.spinner("Processing image..."):
#             description = handle_image_description(uploaded_image.getvalue())
#             st.session_state.image_description = description
#             chat_history.add_user_message("Please describe this image.")
#             chat_history.add_ai_message(description)
#             add_to_chat_history("Please describe this image.", description)

#     if send_button or st.session_state.send_input:
#         if st.session_state.user_question:
#             if "image_description" in st.session_state:
#                 if 'image' in st.session_state.user_question.lower() or 'describe' in st.session_state.user_question.lower():
#                     answer = st.session_state.image_description
#                 else:
#                     answer = llm_chain.run(st.session_state.user_question)
#             else:
#                 answer = llm_chain.run(st.session_state.user_question)

#             add_to_chat_history(st.session_state.user_question, answer)

#             # Display user input and AI response in the container
#             with chat_container:
#                 st.chat_message('user').write(st.session_state.user_question)
#                 st.chat_message('ai').write(answer)
            
#             st.session_state.user_question = ""
#             st.session_state.send_input = False

#     # Display the entire chat history from session state
#     if st.session_state.history:
#         with chat_container:
#             st.write("Chat History:")
#             for message in st.session_state.history:
#                 if isinstance(message, HumanMessage):
#                     st.chat_message("user").write(message.content)
#                 elif isinstance(message, AIMessage):
#                     st.chat_message("ai").write(message.content)

#     save_chat_history()

# if __name__ == '__main__':
#     main()