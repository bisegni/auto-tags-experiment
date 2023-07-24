import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "Salesforce/xgen-7b-8k-inst", 
    trust_remote_code=True,
    )

model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/xgen-7b-8k-inst",
)

def summarize(text):
    header =("header")

    text = header + "### Human: please summarize this: \n\n"+text+"\n###"

    input = tokenizer(text, return_tensor="pt")
    generated_id = model.generate(
        **input,
        max_lenght=2048,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
    )

    summary = tokenizer.decode(
        generated_id[0], 
        skip_special_tokens=True
        ).lstrip()
    summary = summary.split("### Assistant:")
    summary = summary.split("<|endoftext|>")
    return gr.Textbox(value=summary)

with gr.Blocks() as demo:
    with gr.Row():
        text = gr.Textbox(lines=20, label="text")
        summary = gr.Textbox(label="Summary", lines=20)
    submit = gr.Button(text="Summarize")
    submit.click(summarize, inputs=text, outputs=summary)

demo.launch()