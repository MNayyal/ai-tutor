# 🚀 Quick Start Guide: Syrian English Tutor AI

This guide will help you train your AI tutor in just 30-60 minutes using Google Colab.

## ✅ Prerequisites Checklist

- [ ] Google account for Colab access
- [ ] Your `english_textbook.md` file ready
- [ ] 30-60 minutes of time for training

## 🎯 Step-by-Step Instructions

### Step 1: Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account
3. Click "New notebook"

### Step 2: Enable GPU (Important!)

1. Click **Runtime** → **Change runtime type**
2. Select **GPU** from the dropdown
3. Click **Save**

### Step 3: Install Dependencies

Copy and paste this into your first cell and run it:

```python
print("📦 Installing packages...")
!pip install -q torch transformers datasets accelerate peft bitsandbytes gradio

import torch
print(f"✅ GPU available: {torch.cuda.is_available()}")
```

### Step 4: Upload Your Files

Copy and paste this into a new cell:

```python
from google.colab import files
print("📁 Upload your english_textbook.md file:")
uploaded = files.upload()
```

### Step 5: Quick Training Script

Copy this complete training script into a new cell:

```python
# Quick Training Script for Syrian English Tutor
import os, json, torch, re
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import gradio as gr

# Step 1: Extract training data
print("🔄 Processing textbook...")
with open('english_textbook.md', 'r', encoding='utf-8') as f:
    content = f.read()

training_texts = []

# Extract questions
for pattern in [r'(\d+\.)\s*(.+?\?)', r'([A-Z]\.)\s*(.+?\?)']:
    for num, question in re.findall(pattern, content):
        if len(question.strip()) > 10:
            text = f"<|startoftext|>Student: {question.strip()}\nTutor: Let me help you with this question based on the Syrian 12th grade English curriculum.<|endoftext|>"
            training_texts.append(text)

# Extract vocabulary
for word in set(re.findall(r'\*\*(\w+)\*\*', content)):
    if len(word) > 2:
        text = f"<|startoftext|>Student: What does '{word}' mean?\nTutor: The word '{word}' is from your Syrian textbook. Let me explain its meaning and usage.<|endoftext|>"
        training_texts.append(text)

# Add general examples
general_examples = [
    ("Can you help me with Unit 1?", "I'd be happy to help you with Unit 1! Let me guide you through the key topics."),
    ("I need help with grammar.", "Grammar can be challenging, but I'm here to help! Let's work through it step by step."),
    ("How do I prepare for my exam?", "Great question! Let me give you effective study strategies for your English exam."),
]

for question, answer in general_examples:
    text = f"<|startoftext|>Student: {question}\nTutor: {answer}<|endoftext|>"
    training_texts.append(text)

print(f"✅ Created {len(training_texts)} training examples")

# Step 2: Setup model
print("🤖 Loading model...")
MODEL_NAME = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>", "eos_token": "<|endoftext|>",
    "bos_token": "<|startoftext|>", "unk_token": "<|unknown|>"
})

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)
model.resize_token_embeddings(len(tokenizer))

# Step 3: Prepare data
print("📝 Preparing training data...")
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=256)

dataset = Dataset.from_dict({'text': training_texts})
train_test = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test['train'].map(tokenize_function, batched=True, remove_columns=['text'])
eval_dataset = train_test['test'].map(tokenize_function, batched=True, remove_columns=['text'])

# Step 4: Setup LoRA
print("🔧 Setting up efficient training...")
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn", "c_proj"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Step 5: Train
print("🚀 Starting training (this will take 20-40 minutes)...")
training_args = TrainingArguments(
    output_dir="./syrian-tutor", num_train_epochs=2, per_device_train_batch_size=1,
    gradient_accumulation_steps=8, learning_rate=5e-5, fp16=torch.cuda.is_available(),
    gradient_checkpointing=True, logging_steps=25, eval_steps=100, save_steps=200,
    evaluation_strategy="steps", load_best_model_at_end=True, report_to=None
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), tokenizer=tokenizer
)

trainer.train()
trainer.save_model("./syrian-tutor-final")
tokenizer.save_pretrained("./syrian-tutor-final")
print("🎉 Training completed!")
```

### Step 6: Test Your Tutor

Add this to a new cell to test your trained model:

```python
# Test the trained model
print("🧪 Testing your tutor...")
generator = pipeline("text-generation", model="./syrian-tutor-final", tokenizer="./syrian-tutor-final")

def ask_tutor(question):
    prompt = f"<|startoftext|>Student: {question}\nTutor:"
    response = generator(prompt, max_length=150, temperature=0.7, do_sample=True)
    return response[0]['generated_text'].split('Tutor:')[-1].split('<|endoftext|>')[0].strip()

# Test questions
test_questions = [
    "What does 'aspire' mean?",
    "Can you help me with Unit 1?", 
    "I need help with grammar.",
    "How should I prepare for my exam?"
]

for question in test_questions:
    print(f"👨‍🎓 Student: {question}")
    print(f"🎓 Tutor: {ask_tutor(question)}\n")
```

### Step 7: Create Interactive Interface

Finally, add this to create a user-friendly chat interface:

```python
# Create interactive interface
print("🎨 Creating your tutor interface...")

def chat_with_tutor(message, history):
    response = ask_tutor(message)
    history.append([message, response])
    return history, ""

with gr.Blocks(title="Syrian English Tutor") as demo:
    gr.Markdown("# 🎓 Syrian English Tutor AI\nYour personal tutor for Syrian 12th grade English!")
    
    chatbot = gr.Chatbot([], height=400)
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask me anything about English!", scale=9)
        send = gr.Button("Send", scale=1)
    
    send.click(chat_with_tutor, [msg, chatbot], [chatbot, msg])
    msg.submit(chat_with_tutor, [msg, chatbot], [chatbot, msg])

demo.launch(share=True)
print("🚀 Your tutor is ready! Use the interface above to chat with students.")
```

## 🎉 Congratulations!

You now have a working AI tutor! The interface will provide a shareable link that students can use to get help with:

- ✅ Vocabulary explanations
- ✅ Grammar assistance
- ✅ Reading comprehension
- ✅ Writing help
- ✅ Exam preparation

## 💡 Tips for Success

1. **GPU is Essential**: Make sure GPU is enabled in Colab for faster training
2. **Be Patient**: Training takes 20-40 minutes - don't interrupt the process
3. **Test Thoroughly**: Try various questions to see how your tutor responds
4. **Share Responsibly**: The generated link works for 72 hours

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `per_device_train_batch_size` to 1 |
| Training too slow | Ensure GPU is enabled |
| Poor responses | Add more training examples or increase epochs |
| Interface won't load | Check that training completed successfully |

## 📚 What's Next?

- Share the interface link with students
- Collect feedback to improve the tutor
- Consider training on additional materials
- Explore advanced features in the full documentation

---

**Happy teaching! 🎓** Your AI tutor is ready to help Syrian students excel in English!