import base64
import logging
from llama_cpp import Llama  # Assuming this is correctly installed

# Configure logging
logging.basicConfig(level=logging.INFO)

def convert_bytes_to_base64(image_bytes):
    """Convert image bytes to a base64 string."""
    try:
        if not image_bytes:
            logging.error("No image bytes provided for conversion.")
            return None
        encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        return encoded_string
    except Exception as e:
        logging.error(f"Error encoding image to base64: {e}")
        return None

def handle_image(image_bytes, user_message):
    """Handle image input and generate a response based on the image and user message."""
    try:
        # Validate image_bytes is not empty
        if not image_bytes:
            logging.warning("No image data provided.")
            return "No image data provided."
        
        # Validate the message is not empty
        if not user_message:
            logging.warning("No user message provided.")
            return "No user message provided."
        
        # Initialize the Llama model
        logging.info("Initializing the Llama model...")
        llm = Llama(
            model_path="models/llava/ggml-model-q5_k.gguf",  # Adjust to your model path
            n_ctx=1024,
            logits_all=True,
        )

        # Convert image to base64
        image_base64 = convert_bytes_to_base64(image_bytes)
        
        if not image_base64:
            return "Failed to convert image to base64."

        # Formulate message with base64 image string
        message_content = f"data:image/jpeg;base64,{image_base64}\n\n{user_message}"

        # Create chat completion
        logging.info("Creating chat completion...")
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an assistant who describes images."},
                {"role": "user", "content": message_content}
            ]
        )
        
        # Extract and return the response content
        response = output["choices"][0]["message"]["content"]
        logging.info(f"Response from model: {response}")
        return response
    
    except Exception as e:
        logging.error(f"An error occurred while handling the image: {e}")
        return "An error occurred while processing the request."

# Example usage:
# Pass in `image_bytes` and `user_message` to `handle_image` function
