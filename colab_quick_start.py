# 🎓 Syrian English Tutor - Google Colab Quick Start
# ================================================
# Copy and paste each section into separate Colab cells

# CELL 1: Install Dependencies
print("📦 Installing required packages...")
!pip install -q torch transformers datasets accelerate peft bitsandbytes gradio

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling, pipeline
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import gradio as gr
import re
from datetime import datetime

print("✅ All packages installed and imported!")
print(f"🔥 CUDA available: {torch.cuda.is_available()}")

# CELL 2: Upload Files
print("📁 Please upload your english_textbook.md file:")
from google.colab import files
uploaded = files.upload()

if 'english_textbook.md' in uploaded:
    print("✅ Textbook uploaded successfully!")
else:
    print("❌ Please upload english_textbook.md")

# CELL 3: Simple Data Preprocessing
print("🔄 Preprocessing textbook data...")

def create_training_data():
    with open('english_textbook.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    training_texts = []
    
    # Extract questions
    question_patterns = [
        r'(\d+\.)\s*(.+?\?)',
        r'([A-Z]\.)\s*(.+?\?)',
    ]
    
    for pattern in question_patterns:
        matches = re.findall(pattern, content)
        for num, question in matches:
            if len(question.strip()) > 10:
                text = f"<|startoftext|>Student: {question.strip()}\nTutor: Let me help you with this question based on the Syrian 12th grade English curriculum.<|endoftext|>"
                training_texts.append(text)
    
    # Extract vocabulary (words in bold)
    vocab_pattern = r'\*\*(\w+)\*\*\s*(.+?)(?=\n|$)'
    vocab_matches = re.findall(vocab_pattern, content)
    
    for word, definition in vocab_matches:
        if len(word) > 2 and len(definition) > 10:
            text = f"<|startoftext|>Student: What does '{word}' mean?\nTutor: The word '{word}' means: {definition.strip()}<|endoftext|>"
            training_texts.append(text)
    
    # Add general help examples
    general_examples = [
        ("Can you help me with Unit 1?", "I'd be happy to help you with Unit 1! Let me guide you through the key topics."),
        ("I need help with grammar.", "Grammar can be challenging, but I'm here to help! Let's work through it step by step."),
        ("How do I prepare for my exam?", "Great question! Let me give you some effective study strategies for your English exam."),
        ("I'm struggling with vocabulary.", "Vocabulary building is important! Let me help you learn and remember new words effectively."),
        ("Help me with writing.", "I'd love to help you improve your writing! Let's work on structure, grammar, and style."),
    ]
    
    for question, answer in general_examples:
        text = f"<|startoftext|>Student: {question}\nTutor: {answer}<|endoftext|>"
        training_texts.append(text)
    
    print(f"✅ Created {len(training_texts)} training examples")
    return training_texts

training_texts = create_training_data()

# CELL 4: Setup Model and Tokenizer
print("🤖 Setting up model and tokenizer...")

MODEL_NAME = "microsoft/DialoGPT-small"  # Small model for free Colab
print(f"📥 Loading {MODEL_NAME}...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Add special tokens
special_tokens = {
    "pad_token": "<|pad|>",
    "eos_token": "<|endoftext|>",
    "bos_token": "<|startoftext|>",
    "unk_token": "<|unknown|>"
}

tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

model.resize_token_embeddings(len(tokenizer))
print(f"✅ Model loaded: {model.num_parameters():,} parameters")

# CELL 5: Prepare Training Data
print("📝 Preparing training data...")

def tokenize_function(examples):
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    tokenized['labels'] = tokenized['input_ids'].clone()
    return tokenized

# Create and split dataset
dataset = Dataset.from_dict({'text': training_texts})
train_test = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = train_test['train'].map(tokenize_function, batched=True, remove_columns=['text'])
eval_dataset = train_test['test'].map(tokenize_function, batched=True, remove_columns=['text'])

print(f"✅ Training set: {len(train_dataset)} examples")
print(f"✅ Validation set: {len(eval_dataset)} examples")

# CELL 6: Setup LoRA Training
print("🔧 Setting up LoRA for efficient training...")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print(f"📊 Trainable parameters: {model.num_parameters():,}")
print(f"📊 Percentage trainable: {100 * model.num_parameters() / model.base_model.model.num_parameters():.2f}%")

# CELL 7: Training Configuration
print("⚙️ Setting up training configuration...")

training_args = TrainingArguments(
    output_dir="./syrian-tutor",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_steps=50,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    logging_steps=25,
    eval_steps=100,
    save_steps=200,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to=None,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

print("✅ Training configuration ready!")

# CELL 8: Start Training
print("🚀 Starting training...")
print("⏱️ This will take approximately 20-40 minutes...")

try:
    trainer.train()
    trainer.save_model("./syrian-tutor-final")
    tokenizer.save_pretrained("./syrian-tutor-final")
    print("🎉 Training completed successfully!")
except Exception as e:
    print(f"❌ Training failed: {e}")

# CELL 9: Test the Model
print("🧪 Testing the trained model...")

generator = pipeline(
    "text-generation",
    model="./syrian-tutor-final",
    tokenizer="./syrian-tutor-final",
    device=0 if torch.cuda.is_available() else -1
)

def ask_tutor(question):
    prompt = f"<|startoftext|>Student: {question}\nTutor:"
    response = generator(
        prompt,
        max_length=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    full_response = response[0]['generated_text']
    tutor_response = full_response.split('Tutor:')[-1].split('<|endoftext|>')[0].strip()
    return tutor_response

# Test with sample questions
test_questions = [
    "What does 'aspire' mean?",
    "Can you help me with Unit 1?",
    "I need help with grammar.",
    "How should I prepare for my exam?"
]

print("🎯 Testing with sample questions:\n")
for question in test_questions:
    response = ask_tutor(question)
    print(f"👨‍🎓 Student: {question}")
    print(f"🎓 Tutor: {response}\n")
    print("-" * 50)

# CELL 10: Create Interactive Interface
print("🎨 Creating interactive Gradio interface...")

def chat_with_tutor(message, history):
    response = ask_tutor(message)
    history.append([message, response])
    return history, ""

with gr.Blocks(title="Syrian English Tutor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎓 Syrian English Tutor AI
    
    Welcome to your personalized English tutor for the Syrian 12th grade curriculum!
    
    **What I can help you with:**
    - 📚 Vocabulary explanations
    - 📝 Grammar rules and exercises  
    - 📖 Reading comprehension
    - ✍️ Writing assistance
    - 🎯 Exam preparation
    
    **How to use:** Simply type your question below!
    """)
    
    chatbot = gr.Chatbot([], height=400)
    
    with gr.Row():
        msg = gr.Textbox(placeholder="Ask me anything about English! (e.g., 'What does aspire mean?')", scale=9)
        send = gr.Button("Send 📤", scale=1)
    
    with gr.Row():
        clear = gr.Button("Clear Chat 🗑️")
    
    # Example buttons
    gr.Markdown("### 💡 Try these examples:")
    with gr.Row():
        ex1 = gr.Button("What does 'burden' mean?", size="sm")
        ex2 = gr.Button("Help me with Unit 1", size="sm")
        ex3 = gr.Button("Grammar exercises help", size="sm")
        ex4 = gr.Button("How to write essays?", size="sm")
    
    # Event handlers
    send.click(chat_with_tutor, [msg, chatbot], [chatbot, msg])
    msg.submit(chat_with_tutor, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)
    
    ex1.click(lambda: chat_with_tutor("What does 'burden' mean?", []), None, [chatbot, msg])
    ex2.click(lambda: chat_with_tutor("Help me with Unit 1", []), None, [chatbot, msg])
    ex3.click(lambda: chat_with_tutor("I need help with grammar exercises", []), None, [chatbot, msg])
    ex4.click(lambda: chat_with_tutor("How do I write good essays?", []), None, [chatbot, msg])

print("🚀 Launching your Syrian English Tutor interface...")
demo.launch(share=True, debug=True)

# CELL 11: Download Your Trained Model (Optional)
print("💾 Creating download package...")

import shutil
import zipfile

# Create deployment folder
os.makedirs("deployment", exist_ok=True)
shutil.copytree("syrian-tutor-final", "deployment/model", dirs_exist_ok=True)

# Create simple usage script
usage_script = '''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class SyrianTutor:
    def __init__(self, model_path="./model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.generator = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
    
    def ask(self, question):
        prompt = f"<|startoftext|>Student: {question}\\nTutor:"
        response = self.generator(
            prompt, max_length=200, temperature=0.7,
            do_sample=True, pad_token_id=self.tokenizer.eos_token_id
        )
        full_response = response[0]['generated_text']
        return full_response.split('Tutor:')[-1].split('<|endoftext|>')[0].strip()

# Example usage
tutor = SyrianTutor()
while True:
    question = input("Student: ")
    if question.lower() in ['quit', 'exit']: break
    print(f"Tutor: {tutor.ask(question)}")
'''

with open("deployment/tutor.py", "w") as f:
    f.write(usage_script)

with open("deployment/README.md", "w") as f:
    f.write("""
# Syrian English Tutor AI

Your trained AI tutor for Syrian 12th grade English curriculum.

## Usage
1. Install dependencies: `pip install torch transformers`
2. Run: `python tutor.py`
3. Ask questions about English topics!

## Features
- Vocabulary explanations
- Grammar assistance  
- Reading comprehension help
- Writing guidance
- Exam preparation
""")

# Create ZIP file
with zipfile.ZipFile("syrian_tutor.zip", "w") as zipf:
    for root, dirs, files in os.walk("deployment"):
        for file in files:
            zipf.write(os.path.join(root, file), 
                      os.path.relpath(os.path.join(root, file), "deployment"))

print("✅ Model package created!")
print("📥 Downloading...")

from google.colab import files
files.download("syrian_tutor.zip")

print("""
🎉 Congratulations! Your Syrian English Tutor is ready!

📚 What you've accomplished:
✅ Processed the Syrian 12th grade English curriculum
✅ Fine-tuned an AI model for tutoring
✅ Created an interactive interface
✅ Generated a deployment package

🚀 Your tutor can help students with:
📖 Vocabulary and definitions
📝 Grammar explanations  
📚 Reading comprehension
✍️ Writing assistance
🎯 Exam preparation

The interface above is ready to use, and you've downloaded a package 
for local deployment. Happy teaching! 🎓
""")