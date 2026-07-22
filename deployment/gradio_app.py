from __future__ import annotations

import gradio as gr
from transformers import pipeline

MODEL_ID = "yourname/vajra-lm-1b-instruct"
pipe = pipeline("text-generation", model=MODEL_ID, device_map="auto")


def chat(message, history):
    prompt = "".join(f"<|user|>{u}<|assistant|>{a}" for u, a in history) + f"<|user|>{message}<|assistant|>"
    response = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7)[0]["generated_text"]
    return response[len(prompt):]


demo = gr.ChatInterface(fn=chat, title="Foundation LM")

if __name__ == "__main__":
    demo.launch()
