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
        return_tensors="pt",
        max_length=128,
        truncation=True
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=4,
            early_stopping=True
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

css = """
    body, .gradio-container {
        background-color: #1a1a2e !important;
        font-family: 'Georgia', serif;
    }
    
    /* Title area */
    .gradio-container h1 {
        color: #FFD700 !important;
        text-align: center;
        font-size: 2.5em;
        text-shadow: 2px 2px 4px #CC0000, 0 0 20px #FFD700;
        letter-spacing: 2px;
    }
    
    .gradio-container p {
        color: #87CEEB !important;
        text-align: center;
        font-size: 1.1em;
    }

    /* Chat bubbles */
    .message.user {
        background-color: #CC0000 !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
    }
    
    .message.bot {
        background-color: #003A8C !important;
        color: #FFD700 !important;
        border-radius: 18px 18px 18px 4px !important;
        border: 1px solid #FFD700 !important;
    }

    /* Chatbot container */
    .chatbot {
        background-color: #0d0d1a !important;
        border: 2px solid #FFD700 !important;
        border-radius: 12px !important;
    }

    /* Textbox */
    textarea, input[type="text"] {
        background-color: #0d0d1a !important;
        color: #FFD700 !important;
        border: 2px solid #003A8C !important;
        border-radius: 8px !important;
    }
    
    textarea:focus, input[type="text"]:focus {
        border-color: #FFD700 !important;
        box-shadow: 0 0 8px #FFD700 !important;
    }

    /* Submit button */
    button.primary {
        background-color: #CC0000 !important;
        color: #FFD700 !important;
        border: 2px solid #FFD700 !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        letter-spacing: 1px !important;
        transition: all 0.2s ease !important;
    }
    
    button.primary:hover {
        background-color: #FFD700 !important;
        color: #CC0000 !important;
    }

    /* Labels */
    label, .label-wrap {
        color: #87CEEB !important;
    }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# ⚡ Pokémon Move Generator")
    gr.Markdown("Describe a Pokémon move to get its name, type, power, and accuracy!")

    chatbot = gr.Chatbot(label="Chat")
    user_input = gr.Textbox(
        placeholder = "Describe a Pokémon move...",
        label = "Description",
        lines = 2
    )
    submit_btn = gr.Button("Submit", variant="primary")

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