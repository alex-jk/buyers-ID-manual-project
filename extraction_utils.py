import torch
import os
import re

# --- Function to Load a Specific Prompt File ---
def load_system_prompt(prompt_filename):
    """Loads system prompt text from a specified file."""
    system_prompt = ""
    try:
        # Assumes prompt file is in the same directory as this script
        script_dir = os.path.dirname(__file__)
        prompt_path = os.path.join(script_dir, prompt_filename)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        if not system_prompt:
            print(f"WARNING: Prompt file loaded but is empty: {prompt_path}")
            return f"ERROR: Prompt file empty {prompt_filename}" # Return specific error
    except FileNotFoundError:
        print(f"ERROR: Prompt file not found at {prompt_path}")
        return f"ERROR: Prompt file not found {prompt_filename}" # Return specific error
    except Exception as e:
        print(f"ERROR: Could not read prompt file {prompt_path}: {e}")
        return f"ERROR: Could not read prompt file {prompt_filename}" # Return specific error
    return system_prompt

# --- Function to Create Prompt Messages (Now uses loaded prompt) ---
def create_prompt_messages(input_chunk, system_prompt_text):
    """Creates the chat message structure using the provided system prompt text."""

    # Check if system_prompt_text indicates an error from loading
    if "ERROR:" in system_prompt_text:
         print(f"WARNING: Using error message as system prompt: {system_prompt_text}")
         # Optionally, you could return an error flag or raise an exception here too

    # User prompt remains dynamic
    user_prompt = f"""Now, perform the task for the following Input Text:

Input Text:
"{input_chunk}"

Extracted Information:
"""
    messages = [
        {"role": "system", "content": system_prompt_text}, # Use the passed-in system prompt
        {"role": "user", "content": user_prompt},
    ]
    return messages

# --- Function to Extract Information (Now accepts prompt filename) ---
def extract_information(text_chunk, generation_pipeline, prompt_filename):
    """
    Uses the pipeline to extract info based on the prompt specified by prompt_filename.
    """
    if generation_pipeline is None or not callable(generation_pipeline):
        print("Pipeline not initialized or not valid.")
        return "Error: Pipeline not available or invalid."

    # 1. Load the specific system prompt for this task
    system_prompt_text = load_system_prompt(prompt_filename)
    if "ERROR:" in system_prompt_text:
         return f"Error: Failed to load system prompt from '{prompt_filename}'." # Return early if prompt fails

    # 2. Create messages using the loaded system prompt
    messages = create_prompt_messages(text_chunk, system_prompt_text)

    # 3. Call the pipeline and process output (same logic as before)
    try:
        outputs = generation_pipeline(messages, return_full_text=False)

        if outputs and isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict) and 'generated_text' in outputs[0]:
             generated_text = outputs[0]['generated_text'].strip()
        else:
             print(f"Warning: Unexpected output format from pipeline: {outputs}")
             generated_text = "Error: Unexpected pipeline output format."

        if isinstance(generated_text, str): # Ensure it's a string before string methods
            if generated_text.startswith('"') and generated_text.endswith('"'):
                 generated_text = generated_text[1:-1]
            if generated_text.startswith("'") and generated_text.endswith("'"):
                 generated_text = generated_text[1:-1]
            instruction_tail = "Extracted Information:"
            if generated_text.startswith(instruction_tail):
                generated_text = generated_text[len(instruction_tail):].strip()
            if not generated_text or generated_text.lower() == "none":
                return "NONE"

        return generated_text

    except Exception as e:
        print(f"Error during generation pipeline call for {prompt_filename}: {e}")
        return f"Error: Exception during generation. ({type(e).__name__})"

    except Exception as e:
        print(f"Error during generation: {e}")
        return f"Error: Could not generate output. ({e})"

# ----------------- create text chunks
def chunk_text_by_paragraphs(text, min_chunk_chars=100, max_chunk_chars=3000): # Increased max_chunk_chars slightly
    """Splits text into chunks based on paragraphs, trying to stay within size limits."""
    paragraphs = re.split(r'\n\s*\n', text) # Split by double newline
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Check if adding the next paragraph makes the chunk too long
        # Use approximate token length (e.g., 4 chars/token) - conservative estimate
        potential_chunk_len = len(current_chunk) + len(para) + 2 # +2 for newline separators
        if current_chunk and potential_chunk_len > max_chunk_chars:
            # Add current chunk if reasonably long
            if len(current_chunk) >= min_chunk_chars:
                chunks.append(current_chunk)
            # Start new chunk with the current paragraph
            current_chunk = para
        else:
            # Append paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para

    # Add the last remaining chunk
    if current_chunk and len(current_chunk) >= min_chunk_chars:
        chunks.append(current_chunk)

    print(f"Split text into {len(chunks)} chunks.")
    return chunks

# Function to process all chunks with a specific prompt
# Ensure extract_information is imported: from extraction_utils import extract_information
def process_text_chunks_with_prompt(text_chunks, generation_pipeline, prompt_filename_to_use):
    """Processes each text chunk using the extraction pipeline and a specific prompt."""
    all_extractions = []
    if not callable(generation_pipeline):
        print("ERROR: Generation pipeline is not available or not valid.")
        return []
    if not text_chunks:
        print("ERROR: No text chunks provided.")
        return []

    num_chunks = len(text_chunks)
    print(f"Processing {num_chunks} chunks using prompt '{prompt_filename_to_use}'...")

    for i, chunk in enumerate(text_chunks):
        print(f"\n  Processing chunk {i+1}/{num_chunks}. Length: {len(chunk)} characters.")
        # Call the main extraction function from your utils file
        extraction_result = extract_information(chunk, generation_pipeline, prompt_filename_to_use)

        # Only store non-empty/non-"NONE" results
        if extraction_result and extraction_result.strip().upper() != "NONE":
            print(f"    Found relevant info in chunk {i+1}: {extraction_result[:100]}...") # Print start
            all_extractions.append(extraction_result)
        else:
             print(f"    No relevant info found in chunk {i+1}.")
        # Optional delay can go here if needed

    print(f"Finished processing chunks. Found {len(all_extractions)} relevant extractions for this prompt.")
    return all_extractions
