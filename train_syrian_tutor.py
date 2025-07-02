#!/usr/bin/env python3
"""
Syrian English Tutor AI - Training Script
==========================================

This script trains an AI tutor for Syrian 12th grade English curriculum.
Designed to work on Google Colab's free tier.

Author: AI Assistant
Date: 2024
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check if we're in Colab
try:
    import google.colab
    IN_COLAB = True
    print("🔍 Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("🔍 Running locally")

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    packages = [
        "torch",
        "transformers",
        "datasets", 
        "accelerate",
        "peft",
        "bitsandbytes",
        "gradio",
        "markdown",
        "beautifulsoup4"
    ]
    
    if IN_COLAB:
        for package in packages:
            os.system(f"pip install -q {package}")
    
    print("✅ All packages installed!")

def setup_imports():
    """Import all necessary libraries"""
    global AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    global DataCollatorForLanguageModeling, Dataset, LoraConfig, get_peft_model
    global prepare_model_for_kbit_training, pipeline, gr
    
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
        pipeline
    )
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import gradio as gr
    
    print("✅ All imports successful!")

def upload_files():
    """Handle file uploads in Colab"""
    if not IN_COLAB:
        print("⚠️ File upload only available in Google Colab")
        return
    
    from google.colab import files
    
    print("📁 Please upload your files:")
    print("1. english_textbook.md")
    print("2. data_preprocessor.py (if you have it)")
    
    uploaded = files.upload()
    
    if 'english_textbook.md' in uploaded:
        print("✅ Textbook uploaded successfully")
    else:
        print("❌ Please upload english_textbook.md")
        return False
    
    return True

def create_simple_preprocessor():
    """Create a simplified version of the preprocessor if not uploaded"""
    
    preprocessor_code = '''
import re
import json
from typing import List, Dict

class SimpleTextbookPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def extract_content(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def create_training_data(self):
        content = self.extract_content()
        
        # Simple extraction of Q&A patterns
        training_texts = []
        
        # Extract sections with questions
        question_patterns = [
            r'(\d+\.)\s*(.+?\?)',
            r'([A-Z]\.)\s*(.+?\?)',
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, content)
            for num, question in matches:
                if len(question.strip()) > 10:  # Filter short questions
                    training_text = f"<|startoftext|>Student: {question.strip()}\\nTutor: Let me help you with this question. "
                    training_text += "I'll explain this step by step based on the Syrian 12th grade English curriculum.<|endoftext|>"
                    training_texts.append(training_text)
        
        # Extract vocabulary definitions
        vocab_patterns = [
            r'(\*\*\w+\*\*)\s*(.+?)(?=\\n|$)',
            r'(\d+\.\s*\*\*\w+\*\*)\s*(.+?)(?=\\n|$)'
        ]
        
        for pattern in vocab_patterns:
            matches = re.findall(pattern, content)
            for word, definition in matches:
                clean_word = word.strip('*').replace('**', '').strip()
                if len(clean_word) > 2 and len(definition) > 10:
                    training_text = f"<|startoftext|>Student: What does '{clean_word}' mean?\\n"
                    training_text += f"Tutor: The word '{clean_word}' means: {definition.strip()}<|endoftext|>"
                    training_texts.append(training_text)
        
        # Add general help patterns
        general_patterns = [
            "Can you help me with Unit 1?",
            "I need help with grammar exercises.",
            "How do I prepare for my English exam?",
            "Can you explain this reading passage?",
            "I'm struggling with vocabulary.",
            "Help me with writing assignments."
        ]
        
        for question in general_patterns:
            training_text = f"<|startoftext|>Student: {question}\\n"
            training_text += "Tutor: I'd be happy to help you! As your English tutor for the Syrian 12th grade curriculum, "
            training_text += "I can guide you through this topic step by step. Let me explain the key concepts.<|endoftext|>"
            training_texts.append(training_text)
        
        return training_texts
    
    def save_training_data(self, output_file='training_data.json'):
        training_texts = self.create_training_data()
        
        data = {
            "training_texts": training_texts,
            "total_examples": len(training_texts),
            "source": "Syrian 12th Grade English Curriculum"
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created {len(training_texts)} training examples")
        return training_texts
'''
    
    with open('simple_preprocessor.py', 'w') as f:
        f.write(preprocessor_code)
    
    return 'simple_preprocessor.py'

def preprocess_textbook_data():
    """Preprocess the textbook content"""
    print("🔄 Preprocessing textbook data...")
    
    # Try to use uploaded preprocessor, otherwise create simple one
    if os.path.exists('data_preprocessor.py'):
        from data_preprocessor import TextbookPreprocessor
        processor = TextbookPreprocessor("english_textbook.md")
        results = processor.process_textbook()
        
        with open('combined_training_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Format for training
        training_texts = []
        for conv in data['conversations']:
            text = "<|startoftext|>"
            for msg in conv['messages']:
                role = "Student" if msg['role'] == 'user' else "Tutor"
                text += f"{role}: {msg['content']}\n"
            text += "<|endoftext|>"
            training_texts.append(text)
        
        for inst in data['instructions']:
            text = f"<|startoftext|>Student: {inst.get('input', inst['instruction'])}\n"
            text += f"Tutor: {inst['output']}<|endoftext|>"
            training_texts.append(text)
            
    else:
        print("📝 Using simple preprocessor...")
        preprocessor_file = create_simple_preprocessor()
        
        exec(open(preprocessor_file).read())
        processor = SimpleTextbookPreprocessor("english_textbook.md")
        training_texts = processor.save_training_data()
    
    print(f"✅ Generated {len(training_texts)} training examples")
    return training_texts

def setup_model_and_tokenizer():
    """Setup the model and tokenizer"""
    print("🤖 Setting up model and tokenizer...")
    
    # Use a smaller model for free Colab
    MODEL_NAME = "microsoft/DialoGPT-small"  # Smaller for memory efficiency
    
    print(f"📥 Loading {MODEL_NAME}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add special tokens
    special_tokens = {
        "pad_token": "<|pad|>",
        "eos_token": "<|endoftext|>",
        "bos_token": "<|startoftext|>",
        "unk_token": "<|unknown|>"
    }
    
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"📝 Added {num_added} special tokens")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    print(f"✅ Model loaded: {model.num_parameters():,} parameters")
    
    return model, tokenizer

def prepare_training_data(training_texts, tokenizer):
    """Prepare data for training"""
    print("📝 Preparing training data...")
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=256,  # Smaller for memory efficiency
            return_tensors="pt"
        )
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    
    # Create dataset
    dataset = Dataset.from_dict({'text': training_texts})
    
    # Split data
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    
    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    print(f"✅ Training set: {len(train_dataset)} examples")
    print(f"✅ Validation set: {len(eval_dataset)} examples")
    
    return train_dataset, eval_dataset

def setup_lora_training(model):
    """Setup LoRA for efficient training"""
    print("🔧 Setting up LoRA training...")
    
    # LoRA config
    lora_config = LoraConfig(
        r=8,  # Smaller rank for memory efficiency
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Prepare model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    print(f"📊 Trainable parameters: {model.num_parameters():,}")
    total_params = model.base_model.model.num_parameters()
    print(f"📊 Percentage trainable: {100 * model.num_parameters() / total_params:.2f}%")
    
    return model

def train_model(model, tokenizer, train_dataset, eval_dataset):
    """Train the model"""
    print("🚀 Starting training...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./syrian-tutor",
        overwrite_output_dir=True,
        num_train_epochs=2,  # Fewer epochs for quick training
        per_device_train_batch_size=1,  # Very small for memory
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,  # Effective batch size = 8
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
        run_name=f"syrian-tutor-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("⏱️ Training will take approximately 20-40 minutes...")
    
    try:
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model("./syrian-tutor-final")
        tokenizer.save_pretrained("./syrian-tutor-final")
        
        print("🎉 Training completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Try reducing batch size or using a smaller model")
        return False

def test_model(model_path="./syrian-tutor-final"):
    """Test the trained model"""
    print("🧪 Testing the trained model...")
    
    try:
        generator = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        
        test_questions = [
            "What does 'aspire' mean?",
            "Can you help me with Unit 1?",
            "I need help with grammar.",
            "How should I prepare for my exam?"
        ]
        
        print("🎯 Testing with sample questions:\n")
        
        for question in test_questions:
            prompt = f"<|startoftext|>Student: {question}\nTutor:"
            
            response = generator(
                prompt,
                max_length=150,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            full_response = response[0]['generated_text']
            tutor_response = full_response.split('Tutor:')[-1].split('<|endoftext|>')[0].strip()
            
            print(f"👨‍🎓 Student: {question}")
            print(f"🎓 Tutor: {tutor_response}\n")
            print("-" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

def create_gradio_interface(model_path="./syrian-tutor-final"):
    """Create interactive interface"""
    print("🎨 Creating interactive interface...")
    
    try:
        generator = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        
        def chat_with_tutor(message, history):
            prompt = f"<|startoftext|>Student: {message}\nTutor:"
            
            response = generator(
                prompt,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            full_response = response[0]['generated_text']
            tutor_response = full_response.split('Tutor:')[-1].split('<|endoftext|>')[0].strip()
            
            history.append([message, tutor_response])
            return history, ""
        
        # Create interface
        with gr.Blocks(title="Syrian English Tutor") as demo:
            gr.Markdown("# 🎓 Syrian English Tutor AI\n\nYour personal tutor for Syrian 12th grade English!")
            
            chatbot = gr.Chatbot([], height=400)
            
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask me anything about English!", scale=9)
                send = gr.Button("Send", scale=1)
            
            send.click(chat_with_tutor, [msg, chatbot], [chatbot, msg])
            msg.submit(chat_with_tutor, [msg, chatbot], [chatbot, msg])
        
        print("🚀 Launching interface...")
        demo.launch(share=True)
        
    except Exception as e:
        print(f"❌ Interface creation failed: {e}")

def main():
    """Main training pipeline"""
    print("🎓 Syrian English Tutor AI - Training Pipeline")
    print("=" * 50)
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Setup imports
    setup_imports()
    
    # Step 3: Upload files (if in Colab)
    if IN_COLAB:
        if not upload_files():
            return
    else:
        if not os.path.exists('english_textbook.md'):
            print("❌ Please ensure 'english_textbook.md' is in the current directory")
            return
    
    # Step 4: Preprocess data
    training_texts = preprocess_textbook_data()
    
    # Step 5: Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Step 6: Prepare training data
    train_dataset, eval_dataset = prepare_training_data(training_texts, tokenizer)
    
    # Step 7: Setup LoRA
    model = setup_lora_training(model)
    
    # Step 8: Train model
    if train_model(model, tokenizer, train_dataset, eval_dataset):
        # Step 9: Test model
        test_model()
        
        # Step 10: Create interface
        create_gradio_interface()
    
    print("\n🎉 Pipeline completed!")
    print("Your Syrian English Tutor is ready to help students!")

if __name__ == "__main__":
    main()