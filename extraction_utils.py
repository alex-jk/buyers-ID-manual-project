import torch

# --- Define the Prompt (Few-Shot Example) ---
def create_prompt_messages(input_chunk):
    """Creates the chat message structure expected by Phi-3."""
    system_prompt = """You are a research assistant analyzing text for information about buyers and perpetrators in Commercial Sexual Exploitation of Children (CSEC). From the provided "Input Text", extract sentences or key phrases that fall into ANY of the following categories: Buyer Profile, Targeted Offender Characteristics, Trafficker Indicators & Reporting, Offender Indicators & Reporting. Extract only relevant text. If none is found, output "NONE".

Here are some examples:

Input Text:
"Almost half these men are the age 30-39, with the next largest group being men under age 30. The mean age is 33 and the median 31. The youngest survey participant was 18, and the oldest was 67."
Extracted Information:
Almost half these men are the age 30-39, with the next largest group being men under age 30. The mean age is 33 and the median 31. The youngest survey participant was 18, and the oldest was 67.

Input Text:
"The data clearly debunk the myth that CSEC is a problem relegated to the urban core. Men who respond to advertisements for sex with young females come from all over metro Atlanta, the geographic market where the advertisements in this study were targeted."
Extracted Information:
Men who respond to advertisements for sex with young females come from all over metro Atlanta.

Input Text:
"This report also details the economic impact on local communities, unrelated to offender profiles."
Extracted Information:
NONE
"""
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
