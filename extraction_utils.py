import torch
import os
import re
import nltk
nltk.download('punkt')

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
def chunk_text_by_tokens(text, tokenizer, max_chunk_tokens=3000, min_chunk_tokens=50):
    """
    Splits text into chunks by combining complete sentences identified using NLTK,
    ensuring each chunk's token count does not exceed max_chunk_tokens.
    Handles sentences that individually exceed the token limit by truncating at word boundary.
    """
    if not text or not hasattr(tokenizer, 'encode'):
        print("ERROR: Text is empty or tokenizer is invalid/missing.")
        return []

    try:
        nltk.data.find('tokenizers/punkt')
        print("DEBUG: nltk.data.find('tokenizers/punkt') succeeded inside chunk_text_by_tokens.")
        sentences = nltk.sent_tokenize(text)
        print(f"DEBUG: nltk.sent_tokenize produced {len(sentences)} potential sentences.")
    except LookupError:
         print("ERROR: NLTK 'punkt' tokenizer data not found. Ensure nltk.download('punkt') ran successfully.")
         return []
    except Exception as e:
         print(f"ERROR: NLTK sentence tokenization failed: {e}")
         return []

    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if not sent: continue

        try:
            sentence_tokens = len(tokenizer.encode(sent, add_special_tokens=False))
        except Exception as e:
            print(f"    WARNING: Could not tokenize sentence {i+1}. Skipping. Error: {e}")
            continue

        # --- Check if a SINGLE sentence is too long ---
        if sentence_tokens > max_chunk_tokens:
            print(f"    WARNING: Sentence {i+1} (tokens: {sentence_tokens}) alone exceeds max_chunk_tokens ({max_chunk_tokens}). Truncating.")
            # Finalize previous chunk
            if current_chunk_sentences:
                chunk_text = " ".join(current_chunk_sentences)
                final_chunk_tokens = len(tokenizer.encode(chunk_text, add_special_tokens=False))
                if final_chunk_tokens >= min_chunk_tokens: chunks.append(chunk_text)
                else: print(f"    INFO: Dropping previous short chunk (tokens: {final_chunk_tokens}) before oversized sentence.")

            # Truncate at word boundary
            estimated_char_limit = max_chunk_tokens * 3 # Heuristic target
            truncated_sent = sent
            if len(sent) > estimated_char_limit:
                last_space_index = sent.rfind(' ', 0, estimated_char_limit)
                if last_space_index != -1: truncated_sent = sent[:last_space_index]
                else: truncated_sent = sent[:estimated_char_limit] # Fallback

            # Check truncated token count
            truncated_tokens = len(tokenizer.encode(truncated_sent, add_special_tokens=False))
            if truncated_tokens > max_chunk_tokens:
                print(f"    ERROR: Sentence still exceeds token limit ({truncated_tokens}) after truncation. Skipping.")
            elif truncated_tokens >= min_chunk_tokens:
                chunks.append(truncated_sent)
                print(f"    INFO: Added truncated sentence chunk (tokens: {truncated_tokens}).")
            else:
                print(f"    INFO: Truncated sentence too short (tokens: {truncated_tokens}). Skipping.")

            # Reset and continue
            current_chunk_sentences = []
            current_chunk_tokens = 0
            continue
        # ---------------------------------------------

        potential_chunk_tokens = current_chunk_tokens + sentence_tokens + (1 if current_chunk_sentences else 0)

        if current_chunk_sentences and potential_chunk_tokens > max_chunk_tokens:
            # Finalize current chunk
            chunk_text = " ".join(current_chunk_sentences)
            final_chunk_tokens = len(tokenizer.encode(chunk_text, add_special_tokens=False))
            if final_chunk_tokens >= min_chunk_tokens: chunks.append(chunk_text)
            else: print(f"    INFO: Dropping short chunk (tokens: {final_chunk_tokens}).")

            # Start new chunk
            current_chunk_sentences = [sent]
            current_chunk_tokens = sentence_tokens
        else:
            # Add sentence to current chunk
            current_chunk_sentences.append(sent)
            current_chunk_tokens = potential_chunk_tokens # Approximate

    # Add the last chunk
    if current_chunk_sentences:
        chunk_text = " ".join(current_chunk_sentences)
        final_chunk_tokens = len(tokenizer.encode(chunk_text, add_special_tokens=False))
        if final_chunk_tokens >= min_chunk_tokens: chunks.append(chunk_text)
        else: print(f"    INFO: Dropping final short chunk (tokens: {final_chunk_tokens}).")

    print(f"Split text into {len(chunks)} chunks using NLTK sentences (max tokens: {max_chunk_tokens}).")
    return chunks

# --- Function to Process Chunks ---
def process_text_chunks_with_prompt(
    text_chunks,
    generation_pipeline, # This is your 'pipe' object
    prompt_filename_to_use,
    batch_size=8, # Add batch_size control
    max_new_tokens=512
    ):
    """
    Processes text chunks using the generation pipeline and a specific prompt,
    using batching for efficiency. (Minimal update version)

    Args:
        text_chunks (list): A list of strings, where each string is a text chunk.
        generation_pipeline (callable): The Hugging Face pipeline object.
        prompt_filename_to_use (str): The path to the file containing the prompt template.
        batch_size (int): The number of chunks to process simultaneously.

    Returns:
        list: A list of successfully extracted non-empty results.
    """
    all_extractions = []
    if not callable(generation_pipeline):
        print("ERROR: Generation pipeline is not available or not valid.")
        return []
    if not text_chunks:
        print("WARNING: No text chunks provided.")
        return []

    num_chunks = len(text_chunks)
    print(f"\nProcessing {num_chunks} chunks using prompt '{prompt_filename_to_use}' with batch_size={batch_size}...")

    # 1. Read prompt ONCE
    try:
        with open(prompt_filename_to_use, 'r', encoding='utf-8') as pf:
            prompt_template = pf.read().strip()
    except Exception as e:
        print(f"ERROR: Could not read prompt file '{prompt_filename_to_use}': {e}")
        return [f"Error reading prompt {prompt_filename_to_use}" for _ in text_chunks] # Indicate error for all

    # 2. Prepare all inputs (replaces the loop calling extract_information)
    formatted_inputs = []
    print(f"  Formatting {num_chunks} inputs...")
    for chunk in text_chunks:
        # Assume standard formatting: prompt + chunk
        formatted_inputs.append(f"{prompt_template}\n\nText:\n{chunk}")

    # 3. Call pipeline ONCE with the batch
    print(f"  Running pipeline on {len(formatted_inputs)} inputs...")
    try:
        # Use the pipeline directly on the list of inputs
        # Add essential parameters like truncation; adjust max_new_tokens if needed
        results = generation_pipeline(
            formatted_inputs,
            batch_size=batch_size,
            truncation=True,
            max_new_tokens=max_new_tokens, # Adjust as needed, or make it a parameter
            return_full_text=False
            # Add other params like temperature, do_sample if you were using them
        )
        print(f"  Pipeline finished. Received {len(results)} results.")
    except Exception as e:
        print(f"ERROR: Pipeline execution failed during batch processing: {e}")
        # Optionally print traceback
        # import traceback
        # traceback.print_exc()
        return [f"Error during pipeline execution: {e}" for _ in text_chunks] # Indicate error for all

    # 4. Process results (replaces the result checking inside the old loop)
    print(f"  Filtering results...")
    if len(results) != num_chunks:
        print(f"  WARNING: Number of results ({len(results)}) doesn't match number of inputs ({num_chunks}).")

    for i, result_item in enumerate(results):
        # Extract the text - ** ADJUST IF YOUR PIPELINE OUTPUT IS DIFFERENT **
        # Common case: [{'generated_text': '...'}]
        try:
            # Handle potential list-of-lists structure
            if isinstance(result_item, list) and result_item:
                 res_dict = result_item[0]
            elif isinstance(result_item, dict):
                 res_dict = result_item
            else:
                 # Try converting to string as a fallback, or handle specific error
                 extraction_result = str(result_item)
                 res_dict = None # Signal that we couldn't parse as dict

            if res_dict and 'generated_text' in res_dict:
                 extraction_result = res_dict['generated_text'].strip()
            elif not res_dict:
                 # Use the string conversion from above if it wasn't a dict/list[dict]
                 pass # extraction_result already holds str(result_item)
            else:
                 extraction_result = "Error: 'generated_text' not found in result"


        except Exception as e:
            print(f"    Error parsing result for input {i+1}: {e} - Result: {str(result_item)[:100]}...")
            extraction_result = f"Error: Parsing failed - {e}"


        # Apply your original filtering logic
        if extraction_result and extraction_result.strip().upper() != "NONE" and not extraction_result.startswith("Error:"):
            # No need to print here, keep it concise as requested
            all_extractions.append(extraction_result)
        # else: Optional: log discarded results if needed for debugging

    print(f"\nFinished processing batch. Found {len(all_extractions)} successful relevant extractions for '{prompt_filename_to_use}'.")
    return all_extractions
