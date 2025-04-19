import torch
import os

# --- Define path relative to this script file ---
# Assumes prompt_template.txt is in the same directory as extraction_utils.py
PROMPT_TEMPLATE_FILENAME = "prompt_template.txt"
# Construct the full path based on the location of *this* script file
script_dir = os.path.dirname(__file__)
prompt_template_path = os.path.join(script_dir, PROMPT_TEMPLATE_FILENAME)

# --- Load the system prompt from the file ONCE when the module is loaded ---
try:
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT_FROM_FILE = f.read()
except FileNotFoundError:
    print(f"ERROR: Prompt template file not found at {prompt_template_path}")
    SYSTEM_PROMPT_FROM_FILE = "ERROR: Prompt template file not found." # Fallback

# --- Define the Prompt (Few-Shot Example) ---
def create_prompt_messages(input_chunk):
    """Creates the chat message structure expected by Phi-3, loading the system prompt from a file."""

    # Use the system prompt loaded from the file
    system_prompt = SYSTEM_PROMPT_FROM_FILE
    if "ERROR:" in system_prompt: # Check if loading failed
         print("WARNING: Using fallback/error message as system prompt due to file loading issue.")

    # User prompt remains dynamic
    user_prompt = f"""Now, perform the task for the following Input Text:

Input Text:
"{input_chunk}"

Extracted Information:
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages

# --- Function to Extract Information ---
def extract_information(text_chunk, generation_pipeline):
    """
    Uses the loaded model pipeline to extract information based on the prompt.
    """
    if generation_pipeline is None:
        print("Pipeline not initialized. Please ensure it's created and passed correctly.")
        return "Error: Pipeline not available."

    messages = create_prompt_messages(text_chunk)

    try:
        outputs = generation_pipeline(messages, return_full_text=False)

        if outputs and isinstance(outputs, list) and len(outputs) > 0 and 'generated_text' in outputs[0]:
             generated_text = outputs[0]['generated_text'].strip()
        else:
             print(f"Warning: Unexpected output format from pipeline: {outputs}")
             generated_text = "Error: Unexpected pipeline output format."
             return generated_text # Return error early

        # Basic cleanup
        if generated_text.startswith('"') and generated_text.endswith('"'):
             generated_text = generated_text[1:-1]
        if generated_text.startswith("'") and generated_text.endswith("'"):
             generated_text = generated_text[1:-1]
        instruction_tail = "Extracted Information:"
        if generated_text.startswith(instruction_tail):
            generated_text = generated_text[len(instruction_tail):].strip()

        return generated_text

    except Exception as e:
        print(f"Error during generation: {e}")
        return f"Error: Could not generate output. ({e})"
