
# Fine-tuning an LLM with QLoRA

A hands-on implementation of QLoRA fine-tuning on TinyLlama-1.1B for medical Q&A.
This is a learning project demonstrating the full fine-tuning pipeline — not a production chatbot.

## What I learned
- LoRA and QLoRA theory (low-rank adaptation, 4-bit NF4 quantization)
- Fine-tuning with PEFT and TRL (SFTTrainer)
- Dataset preparation in Alpaca instruction format
- Deploying a fine-tuned model with Gradio (temporary public URL via Kaggle GPU)
  
## Repository Structure
- `training.ipynb` — full fine-tuning pipeline (data prep, QLoRA config, training, upload to HF)
- `medical-chatbot-inference.ipynb` — load adapters and run inference with Gradio
  
## Stack
- **Base model:** TinyLlama-1.1B-Chat-v1.0
- **Technique:** QLoRA (4-bit NF4, r=16)
- **Training data:** 1,000 medical Q&A examples (MedQuad + custom)
- **Tools:** Transformers, PEFT, BitsAndBytes, TRL, Gradio
- **Platform:** Kaggle (Tesla T4 GPU)

## Results
- Trainable parameters: 2.25M (0.2% of total)
- Training loss reduced from ~2.1 → ~1.44 over 3 epochs
- Live inference via Gradio interface on Kaggle GPU

## Links
- 🤗 Adapters: [SafiaSidimoussa/medical-chatbot-tinyllama](https://huggingface.co/SafiaSidimoussa/medical-chatbot-tinyllama)
