import time
import os
os.environ['HF_HOME'] = "D:/RFP_Finetuning/hf_cache"


from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch
from dotenv import load_dotenv


load_dotenv()


if torch.cuda.is_available():
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
    available_memory = total_memory - allocated_memory
    print(f"Total GPU memory: {total_memory:.1f} GB")
    print(f"Available GPU memory: {available_memory:.1f} GB")

MODEL = os.environ['VISION_MODEL']
# Load the model and processor
processor = AutoProcessor.from_pretrained(
    MODEL,
    trust_remote_code=True
)
model = AutoModelForImageTextToText.from_pretrained(
MODEL,
    torch_dtype=torch.float16,  # Reduce memory usage
    device_map="auto",         # Automatically distribute across available devices
    # low_cpu_mem_usage=True,   # Reduce CPU memory usage during loading
    trust_remote_code=True
)

def extract_text_from_local_image(image_path, prompt="Please extract and return all the text visible in this image and covert to English language if the input is in other language."):
    """
    Extract text from a local image file using the vision-language model.
    
    Args:
        image_path (str): Path to the local image file
        prompt (str): Text prompt to instruct the model what to do with the image
    
    Returns:
        str: Extracted text from the image
    """
    # Verify that the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Open and load the local image
    image = Image.open(image_path)
    
    # Create the message structure with local image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # Use local image object instead of URL
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Process the input
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    # Generate the response
    outputs = model.generate(
        **inputs, 
        max_new_tokens=1024,  # Increased for potentially longer text extraction
        do_sample=False,     # Deterministic output for text extraction
    )
    
    # Decode and return the generated text
    generated_text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return generated_text.strip()

# Example usage
if __name__ == "__main__":
    # Specify the path to your local image file
    image_path = "arabic.jpg"

    try:
        start_time = time.perf_counter()
        # Extract text from the local image
        extracted_text = extract_text_from_local_image(image_path)
        end_time = time.perf_counter()


        print(f"Text extraction time :-" , end_time - start_time)
        
        # Print the extracted text
        print("Extracted Text:")
        print("=" * 50)
        print(extracted_text)
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred during text extraction: {e}")

    

    