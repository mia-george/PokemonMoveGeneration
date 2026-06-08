import torch
import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("saved_model")
model = T5ForConditionalGeneration.from_pretrained("saved_model")
model = model.to(device)
model.eval()

def generate_move(user_input, history):
    input_text = "predict move: " + user_input

    inputs = tokenizer(
        input_text,
        return_tensors = "pt",
        max_length = 128,
        truncation = True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"],
            max_length = 64,
            num_beams = 4,
            early_stopping = True
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    parsed = {}
    parts = result.split(" | ")
    for part in parts:
        if ": " in part:
            key, value = part.split(": ", 1)
            parsed[key.strip()] = value.strip()

    if not parsed:
        response = "Sorry, I couldn't find that move. Try describing it differently."
    else:
        response = (
            f"**Move:** {parsed.get('move', 'N/A').title()}\n"
            f"**Type:** {parsed.get('type', 'N/A').title()}\n"
            f"**Power:** {parsed.get('power', 'N/A')}\n"
            f"**Accuracy:** {parsed.get('accuracy', 'N/A')}"
        )

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})
    return history, ""

with gr.Blocks() as demo:
    gr.Markdown("# Pokemon Move Generator")
    gr.Markdown("Describe a Pokemon move, and I'll generate it for you.")
    
    chatbot = gr.Chatbot(label="Chat")
    user_input = gr.Textbox(
        placeholder = "Describe a Pokemon move...",
        label = "Your Description",
        lines = 2
    )
    submit_btn = gr.Button("Submit")

    submit_btn.click(
        fn = generate_move,
        inputs = [user_input, chatbot],
        outputs = [chatbot, user_input]
    )
    user_input.submit(
        fn = generate_move,
        inputs = [user_input, chatbot],
        outputs = [chatbot, user_input]
    )

demo.launch(share=True)