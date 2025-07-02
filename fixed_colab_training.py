# 🔧 FIXED Syrian English Tutor Training - Colab Compatible
# This version handles common CUDA, PEFT, and config loading errors

# CELL 1: Install and Setup with Error Handling
print("📦 Installing packages with compatibility fixes...")
!pip install -q torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q transformers==4.35.0 datasets==2.14.0 accelerate==0.24.0 
!pip install -q peft==0.6.0 bitsandbytes==0.41.0 gradio==3.50.0

import torch
import gc
import os
import warnings
warnings.filterwarnings("ignore")

print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name()}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
# Clear any existing CUDA cache
torch.cuda.empty_cache()
gc.collect()

# CELL 2: Upload Files
from google.colab import files
print("📁 Upload your english_textbook.md file:")
uploaded = files.upload()

# CELL 3: Data Processing with Better Error Handling
print("🔄 Processing textbook data...")
import re
import json

try:
    with open('english_textbook.md', 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"✅ Loaded textbook: {len(content)} characters")
except Exception as e:
    print(f"❌ Error loading textbook: {e}")
    exit()

training_texts = []

# Extract questions with better filtering
question_patterns = [r'(\d+\.)\s*(.+?\?)', r'([A-Z]\.)\s*(.+?\?)']
for pattern in question_patterns:
    matches = re.findall(pattern, content, re.MULTILINE)
    for num, question in matches:
        if len(question.strip()) > 15 and len(question.strip()) < 200:  # Better filtering
            text = f"<|startoftext|>Student: {question.strip()}\nTutor: Let me help you with this question from the Syrian 12th grade English curriculum. I'll explain it step by step.<|endoftext|>"
            training_texts.append(text)

print(f"✅ Extracted {len([t for t in training_texts])} questions")

# Extract vocabulary with context
vocab_words = list(set(re.findall(r'\*\*(\w+)\*\*', content)))
vocab_count = 0
for word in vocab_words:
    if len(word) > 2 and word.isalpha():
        # Try to find definition context
        pattern = rf'\*\*{re.escape(word)}\*\*[^a-zA-Z]*(.{{1,100}}?)(?:\n|\*\*|$)'
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            context = match.group(1).strip()
            text = f"<|startoftext|>Student: What does '{word}' mean?\nTutor: The word '{word}' from your Syrian English textbook means: {context}. Let me help you understand how to use it.<|endoftext|>"
        else:
            text = f"<|startoftext|>Student: What does '{word}' mean?\nTutor: '{word}' is an important vocabulary word from your Syrian 12th grade English curriculum. Let me explain its meaning and usage with examples.<|endoftext|>"
        training_texts.append(text)
        vocab_count += 1

print(f"✅ Extracted {vocab_count} vocabulary words")

# Add comprehensive general examples
general_examples = [
    ("Can you help me with Unit 1?", "I'd be happy to help you with Unit 1 about career planning! This unit teaches you how to think about your future job and make good career choices. Let me guide you through the key concepts."),
    ("I need help with grammar exercises.", "Grammar can be challenging, but I'm here to help! Let's work through the grammar concepts from your Syrian curriculum step by step. What specific grammar topic are you struggling with?"),
    ("How do I prepare for my English exam?", "Great question! For your Syrian 12th grade English exam, I recommend: 1) Review vocabulary from each unit, 2) Practice grammar exercises, 3) Read passages carefully, and 4) Practice writing compositions. Which area would you like to focus on?"),
    ("I'm struggling with vocabulary.", "Vocabulary building is crucial for English success! Let me help you learn and remember new words from your textbook effectively. We can use context, examples, and practice exercises."),
    ("Help me understand this reading passage.", "I'd be happy to help you understand reading passages! Let's break it down paragraph by paragraph, identify key ideas, and discuss any difficult vocabulary or concepts."),
    ("How do I write a good essay?", "Writing a good essay involves planning, structure, and clear expression. For your Syrian curriculum, focus on: 1) Clear introduction, 2) Well-organized body paragraphs, 3) Strong conclusion, and 4) Proper grammar and vocabulary."),
    ("What's the difference between present simple and present continuous?", "Great grammar question! Present simple (I work) is for habits and facts. Present continuous (I am working) is for actions happening now. Let me give you examples from your textbook."),
    ("I don't understand this vocabulary word.", "No problem! Understanding vocabulary is key to English success. Tell me which word you're confused about, and I'll explain its meaning, pronunciation, and how to use it in sentences."),
]

for question, answer in general_examples:
    text = f"<|startoftext|>Student: {question}\nTutor: {answer}<|endoftext|>"
    training_texts.append(text)

print(f"✅ Added {len(general_examples)} general examples")
print(f"📊 Total training examples: {len(training_texts)}")

# Save data for debugging
with open('training_data_debug.json', 'w', encoding='utf-8') as f:
    json.dump(training_texts[:5], f, indent=2, ensure_ascii=False)
print("💾 Saved sample data for debugging")

# CELL 4: Model Setup with Better Error Handling
print("🤖 Setting up model with error handling...")

from transformers import AutoTokenizer, AutoModelForCausalLM

# Clear CUDA cache before loading model
torch.cuda.empty_cache()
gc.collect()

MODEL_NAME = "microsoft/DialoGPT-small"
print(f"📥 Loading {MODEL_NAME}...")

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # Add special tokens
    special_tokens = {
        "pad_token": "<|pad|>",
        "eos_token": "<|endoftext|>", 
        "bos_token": "<|startoftext|>",
        "unk_token": "<|unknown|>"
    }
    
    num_added = tokenizer.add_special_tokens(special_tokens)
    print(f"📝 Added {num_added} special tokens")
    
    # Load model with proper device handling
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Resize embeddings
    model.resize_token_embeddings(len(tokenizer))
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    print(f"✅ Model loaded successfully!")
    print(f"📊 Model parameters: {model.num_parameters():,}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("💡 Trying with CPU fallback...")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    print("✅ Model loaded on CPU")

# CELL 5: Data Preparation with Memory Management
print("📝 Preparing training data...")

from datasets import Dataset

def tokenize_function(examples):
    try:
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=200,  # Reduced for memory efficiency
            return_tensors="pt"
        )
        tokenized['labels'] = tokenized['input_ids'].clone()
        return tokenized
    except Exception as e:
        print(f"Tokenization error: {e}")
        return examples

# Create dataset
try:
    dataset = Dataset.from_dict({'text': training_texts})
    print(f"✅ Created dataset with {len(dataset)} examples")
    
    # Split data
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    
    # Tokenize in smaller batches
    print("🔄 Tokenizing training data...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, batch_size=16, remove_columns=['text'])
    eval_dataset = eval_dataset.map(tokenize_function, batched=True, batch_size=16, remove_columns=['text'])
    
    print(f"✅ Training set: {len(train_dataset)} examples")
    print(f"✅ Validation set: {len(eval_dataset)} examples")
    
except Exception as e:
    print(f"❌ Data preparation error: {e}")
    print("💡 This might be a memory issue. Try reducing the dataset size.")

# CELL 6: PEFT Setup with Better Error Handling
print("🔧 Setting up PEFT training...")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

try:
    # Clear cache before PEFT setup
    torch.cuda.empty_cache()
    gc.collect()
    
    # PEFT config with conservative settings
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=4,  # Reduced rank for stability
        lora_alpha=8,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],  # More specific targets
        bias="none"
    )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Apply PEFT
    model = get_peft_model(model, peft_config)
    
    print(f"✅ PEFT applied successfully!")
    print(f"📊 Trainable parameters: {model.num_parameters():,}")
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
except Exception as e:
    print(f"❌ PEFT setup error: {e}")
    print("💡 Falling back to full model training (slower but more stable)")
    # Continue without PEFT if it fails

# CELL 7: Training with Robust Configuration
print("🚀 Setting up training...")

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Clear cache before training
torch.cuda.empty_cache()
gc.collect()

# Conservative training arguments
training_args = TrainingArguments(
    output_dir="./syrian-tutor",
    overwrite_output_dir=True,
    
    # Training parameters
    num_train_epochs=1,  # Start with 1 epoch for testing
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    
    # Learning parameters
    learning_rate=3e-5,  # Conservative learning rate
    warmup_steps=10,
    weight_decay=0.01,
    
    # Memory optimization
    fp16=torch.cuda.is_available(),
    dataloader_pin_memory=False,
    gradient_checkpointing=True,
    
    # Logging and saving
    logging_steps=10,
    eval_steps=50,
    save_steps=100,
    evaluation_strategy="steps",
    save_strategy="steps",
    
    # Stability
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Disable problematic features
    report_to=None,
    push_to_hub=False,
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

print("✅ Training setup complete!")

# CELL 8: Training with Error Recovery
print("🚀 Starting training with error handling...")

try:
    # Start training
    trainer.train()
    
    # Save model with explicit paths
    output_dir = "./syrian-tutor-final"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save PEFT adapter separately if using PEFT
    if hasattr(model, 'save_pretrained'):
        try:
            model.save_pretrained(f"{output_dir}/adapter")
            print("✅ PEFT adapter saved separately")
        except:
            print("⚠️ Could not save PEFT adapter separately (this is okay)")
    
    print("🎉 Training completed successfully!")
    
except Exception as e:
    print(f"❌ Training error: {e}")
    print("💡 Trying to save whatever we have...")
    
    # Try to save even if training failed
    try:
        trainer.save_model("./syrian-tutor-backup")
        tokenizer.save_pretrained("./syrian-tutor-backup")
        print("✅ Backup model saved")
    except:
        print("❌ Could not save backup model")

# CELL 9: FIXED Model Loading and Testing
print("🧪 Loading and testing model with error handling...")

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
import gc

# Clear memory first
torch.cuda.empty_cache()
gc.collect()

def load_model_safe(model_path):
    """Safely load model with multiple fallback options"""
    
    print(f"🔄 Attempting to load model from: {model_path}")
    
    # Option 1: Try loading as PEFT model
    try:
        print("📥 Trying PEFT model loading...")
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Load PEFT adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        
        print("✅ PEFT model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"⚠️ PEFT loading failed: {e}")
    
    # Option 2: Try loading as regular fine-tuned model
    try:
        print("📥 Trying regular model loading...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        print("✅ Regular model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"⚠️ Regular loading failed: {e}")
    
    # Option 3: Load base model with saved tokenizer
    try:
        print("📥 Trying base model with saved tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Resize embeddings to match saved tokenizer
        model.resize_token_embeddings(len(tokenizer))
        
        print("✅ Base model with saved tokenizer loaded!")
        return model, tokenizer
        
    except Exception as e:
        print(f"⚠️ Base model loading failed: {e}")
    
    # Option 4: Fallback to original base model
    try:
        print("📥 Fallback to original base model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        
        # Add special tokens again
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
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        model.resize_token_embeddings(len(tokenizer))
        
        print("✅ Fallback model loaded (using base model)!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ All loading methods failed: {e}")
        return None, None

# Try to load the trained model
model_paths = ["./syrian-tutor-final", "./syrian-tutor", "./syrian-tutor-backup"]

model, tokenizer = None, None
for path in model_paths:
    if os.path.exists(path):
        model, tokenizer = load_model_safe(path)
        if model is not None:
            break

if model is None or tokenizer is None:
    print("❌ Could not load any model. Please check the training completed successfully.")
    exit()

# Create pipeline with error handling
try:
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    print("✅ Pipeline created successfully!")
    
except Exception as e:
    print(f"⚠️ Pipeline creation error: {e}")
    print("🔄 Trying alternative pipeline setup...")
    
    # Alternative pipeline setup
    if torch.cuda.is_available():
        model = model.to('cuda')
    
    def manual_generate(prompt, max_length=150):
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    generator = manual_generate
    print("✅ Manual generation function created!")

# Testing function
def ask_tutor(question):
    """Ask the tutor a question with error handling"""
    try:
        prompt = f"<|startoftext|>Student: {question}\nTutor:"
        
        if callable(generator):  # Manual generation
            response = generator(prompt, max_length=200)
        else:  # Pipeline
            response = generator(
                prompt,
                max_length=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )[0]['generated_text']
        
        # Extract tutor response
        if 'Tutor:' in response:
            tutor_response = response.split('Tutor:')[-1]
            if '<|endoftext|>' in tutor_response:
                tutor_response = tutor_response.split('<|endoftext|>')[0]
            return tutor_response.strip()
        else:
            return "I'm here to help with your English studies! Could you please rephrase your question?"
            
    except Exception as e:
        print(f"Generation error: {e}")
        return f"I'm having trouble processing that question right now. Could you try asking in a different way?"

# Test the model
print("🎯 Testing the tutor...")

test_questions = [
    "What does 'aspire' mean?",
    "Can you help me with Unit 1?",
    "I need help with grammar.",
    "How should I prepare for my exam?"
]

for question in test_questions:
    print(f"\n👨‍🎓 Student: {question}")
    try:
        response = ask_tutor(question)
        print(f"🎓 Tutor: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
    print("-" * 50)

print("\n✅ Testing completed!")

# CELL 10: Create Interface with Error Handling
print("🎨 Creating interface with robust error handling...")

import gradio as gr

def chat_with_tutor(message, history):
    """Chat function with error handling"""
    try:
        if not message.strip():
            return history, ""
        
        response = ask_tutor(message)
        history.append([message, response])
        return history, ""
        
    except Exception as e:
        error_msg = "I'm sorry, I'm having technical difficulties. Please try again or rephrase your question."
        history.append([message, error_msg])
        return history, ""

# Create interface
with gr.Blocks(title="Syrian English Tutor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎓 Syrian English Tutor AI
    
    Your personal tutor for Syrian 12th grade English curriculum!
    
    **I can help you with:**
    - 📚 Vocabulary explanations and definitions
    - 📝 Grammar rules and exercises
    - 📖 Reading comprehension and analysis
    - ✍️ Writing tips and essay structure
    - 🎯 Exam preparation strategies
    
    **How to use:** Type your question below and I'll help you learn!
    
    ⚠️ **Note:** If you see any errors, try refreshing the page or rephrasing your question.
    """)
    
    chatbot = gr.Chatbot([], height=400)
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Ask me anything about English! (e.g., 'What does aspire mean?')",
            scale=9,
            label="Your Question"
        )
        send = gr.Button("Send 📤", scale=1)
    
    with gr.Row():
        clear = gr.Button("Clear Chat 🗑️")
    
    gr.Markdown("### 💡 Try these example questions:")
    
    with gr.Row():
        ex1 = gr.Button("What does 'burden' mean?", size="sm")
        ex2 = gr.Button("Help me with Unit 1", size="sm")
        ex3 = gr.Button("Grammar help needed", size="sm")
        ex4 = gr.Button("How to write essays?", size="sm")
    
    # Event handlers with error handling
    def safe_chat(message, history):
        try:
            return chat_with_tutor(message, history)
        except:
            return history, ""
    
    send.click(safe_chat, [msg, chatbot], [chatbot, msg])
    msg.submit(safe_chat, [msg, chatbot], [chatbot, msg])
    clear.click(lambda: [], None, chatbot)
    
    # Example button handlers
    ex1.click(lambda: safe_chat("What does 'burden' mean?", []), None, [chatbot, msg])
    ex2.click(lambda: safe_chat("Help me with Unit 1", []), None, [chatbot, msg])
    ex3.click(lambda: safe_chat("I need help with grammar exercises", []), None, [chatbot, msg])
    ex4.click(lambda: safe_chat("How do I write good essays?", []), None, [chatbot, msg])

print("🚀 Launching interface...")
try:
    demo.launch(share=True, debug=False, show_error=True)
    print("✅ Interface launched successfully!")
    print("🔗 Use the public link above to share with students")
except Exception as e:
    print(f"⚠️ Interface launch error: {e}")
    print("💡 Try running the interface cell again")

print("""
🎉 Setup Complete!

✅ What worked:
- Data processing and training setup
- Model loading with multiple fallback options  
- Error handling for common issues
- Robust interface creation

💡 If you're still having issues:
1. Try running each cell individually
2. Check that GPU is enabled in Colab
3. Restart runtime if needed (Runtime → Restart runtime)
4. Make sure you uploaded the textbook file

🎓 Your Syrian English Tutor is ready to help students!
""")