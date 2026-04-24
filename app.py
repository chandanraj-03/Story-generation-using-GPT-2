import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = "./model"

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate function
def generate_story(prompt):
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Story:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Story:" in result:
        result = result.split("### Story:")[-1].strip()

    return result

# Gradio UI
interface = gr.Interface(
    fn=generate_story,
    inputs=gr.Textbox(lines=3, placeholder="Enter your story prompt..."),
    outputs=gr.Textbox(lines=10),
    title="AI Story Generator",
    description="Generate creative stories using a fine-tuned GPT-2 model"
)

if __name__ == "__main__":
    interface.launch()
