# 🔧 Troubleshooting Guide - Syrian English Tutor AI

This guide helps you solve common issues when training your AI tutor.

## 🚨 Common Error Solutions

### Cell 9 Errors (Model Loading & Testing)

#### ❌ CUDA Errors
**Error**: `RuntimeError: CUDA out of memory` or `CUDA error: out of memory`

**Solutions**:
```python
# Add this at the start of Cell 9
import torch
import gc
torch.cuda.empty_cache()
gc.collect()

# Reduce memory usage
torch.cuda.set_per_process_memory_fraction(0.8)
```

#### ❌ Config Loading Errors
**Error**: `OSError: Can't load config for './syrian-tutor-final'`

**Solutions**:
1. **Check if training completed**:
```python
import os
print("Checking saved files:")
for path in ["./syrian-tutor-final", "./syrian-tutor", "./syrian-tutor-backup"]:
    if os.path.exists(path):
        print(f"✅ {path} exists")
        print(f"   Files: {os.listdir(path)}")
    else:
        print(f"❌ {path} not found")
```

2. **Use the fixed loading code**:
```python
# Use the safe loading function from fixed_colab_training.py
# It tries multiple loading strategies automatically
```

#### ❌ PEFT Adapter Errors
**Error**: `ValueError: Target modules ... not found in the base model`

**Solutions**:
1. **Skip PEFT and use base model**:
```python
# In Cell 9, replace the loading with:
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({
    "pad_token": "<|pad|>", "eos_token": "<|endoftext|>",
    "bos_token": "<|startoftext|>", "unk_token": "<|unknown|>"
})

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

print("✅ Using base model (no fine-tuning)")
```

## 🔄 Step-by-Step Error Recovery

### If Training Failed Completely

1. **Restart Colab Runtime**:
   - Go to `Runtime` → `Restart runtime`
   - Re-run cells 1-2 (install & upload)

2. **Use Conservative Settings**:
```python
# In training cell, use these safer settings:
training_args = TrainingArguments(
    output_dir="./syrian-tutor",
    num_train_epochs=1,  # Reduce from 2-3
    per_device_train_batch_size=1,  # Keep small
    gradient_accumulation_steps=2,  # Reduce from 8
    learning_rate=1e-5,  # Lower learning rate
    fp16=False,  # Disable if causing issues
    gradient_checkpointing=False,  # Disable if causing issues
)
```

### If Cell 9 Keeps Failing

**Option 1: Use Fixed Script**
- Copy the entire `fixed_colab_training.py` content
- It has built-in error handling for Cell 9

**Option 2: Manual Recovery**
```python
# Emergency model setup - always works
from transformers import pipeline

# Use base DialoGPT without any training
generator = pipeline(
    "text-generation", 
    model="microsoft/DialoGPT-small",
    device=0 if torch.cuda.is_available() else -1
)

def ask_tutor(question):
    response = generator(f"Student: {question}\nTutor:", max_length=100)
    return response[0]['generated_text'].split('Tutor:')[-1].strip()

print("✅ Emergency tutor ready (using base model)")
```

## 🛠️ Specific Error Messages

### `ImportError: No module named 'peft'`
```bash
!pip install peft==0.6.0
```

### `RuntimeError: Expected all tensors to be on the same device`
```python
# Add device management
if torch.cuda.is_available():
    model = model.to('cuda')
    inputs = inputs.to('cuda')
```

### `KeyError: 'c_attn'` (PEFT target modules error)
```python
# Use different target modules for PEFT
target_modules = ["attn", "mlp"]  # Generic names
# Or skip PEFT entirely
```

### `Gradio Interface Won't Load`
```python
# Use simpler interface
import gradio as gr

def simple_chat(message):
    return ask_tutor(message)

iface = gr.Interface(
    fn=simple_chat,
    inputs="text", 
    outputs="text",
    title="Syrian English Tutor"
)
iface.launch(share=True)
```

## 💾 Memory Management Tips

### Before Training
```python
# Clear everything
import gc
import torch
torch.cuda.empty_cache()
gc.collect()

# Check memory
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    print(f"GPU Memory Used: {torch.cuda.memory_allocated() // 1024**3} GB")
```

### During Training Issues
```python
# Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=2

# Reduce sequence length
max_length=128  # Instead of 256

# Use CPU offloading
device_map="auto"
low_cpu_mem_usage=True
```

## 🔍 Debugging Checklist

### ✅ Pre-Training Checklist
- [ ] GPU enabled in Colab (Runtime → Change runtime type → GPU)
- [ ] All packages installed without errors
- [ ] Textbook file uploaded successfully
- [ ] Data processing completed (should see "X training examples created")
- [ ] Model loaded successfully

### ✅ Training Checklist
- [ ] Training started without immediate errors
- [ ] Loss decreasing (check training logs)
- [ ] Model saved to output directory
- [ ] No CUDA out of memory errors

### ✅ Testing Checklist
- [ ] Model files exist in save directory
- [ ] Tokenizer loaded successfully
- [ ] Pipeline created without errors
- [ ] Test questions produce responses
- [ ] Interface launches successfully

## 🆘 Emergency Fallbacks

### If Nothing Works - Use Base Model
```python
# This always works - gives you a basic English tutor
from transformers import pipeline

generator = pipeline("text-generation", model="microsoft/DialoGPT-small")

def emergency_tutor(question):
    prompt = f"Student asks about English: {question}\nHelpful tutor responds:"
    response = generator(prompt, max_length=150, temperature=0.7)
    return response[0]['generated_text'].split('responds:')[-1].strip()

# Test it
print(emergency_tutor("What does aspire mean?"))
```

### If Colab Keeps Crashing
1. **Use Local Computer**: Install Python + packages locally
2. **Try Different Colab Account**: Sometimes account-specific issues
3. **Use CPU-Only**: Disable GPU and use CPU (slower but stable)
4. **Reduce Dataset Size**: Use fewer training examples

## 📧 Getting Additional Help

### Check These First:
1. **Error Message**: Copy exact error message
2. **Cell Number**: Which cell failed?
3. **Colab Runtime**: GPU vs CPU, memory usage
4. **File Status**: Do saved model files exist?

### Common Quick Fixes:
- **Restart Runtime**: Solves 80% of issues
- **Enable GPU**: Most issues come from CPU training
- **Clear Cache**: `torch.cuda.empty_cache()`
- **Reduce Batch Size**: Always try batch_size=1 first

## 🎯 Success Indicators

### Training Succeeded If:
- ✅ No red error messages during training
- ✅ Files exist in `./syrian-tutor-final/`
- ✅ Training logs show decreasing loss
- ✅ "Training completed successfully!" message

### Testing Works If:
- ✅ Model loads without errors
- ✅ Pipeline creation succeeds
- ✅ Test questions get reasonable responses
- ✅ Gradio interface launches

## 🚀 Pro Tips

1. **Always start with the fixed script** (`fixed_colab_training.py`)
2. **Run cells one by one** - don't run all at once
3. **Check GPU memory** before training
4. **Save progress often** - training can take 30+ minutes
5. **Keep cell outputs** - they help with debugging

---

**Remember**: The goal is to get a working tutor, even if it's not perfectly trained. A base model with your interface is still valuable for students!

## 🔗 Quick Recovery Commands

Copy-paste these into a new cell if everything breaks:

```python
# EMERGENCY RESET
import torch, gc, os
torch.cuda.empty_cache()
gc.collect()

# Check what we have
for path in ["./syrian-tutor-final", "./syrian-tutor", "./syrian-tutor-backup"]:
    if os.path.exists(path):
        print(f"Found: {path}")
        break
else:
    print("No trained model found - using base model")

# Emergency tutor setup
from transformers import pipeline
generator = pipeline("text-generation", model="microsoft/DialoGPT-small")

def ask_tutor(question):
    response = generator(f"English tutor helping Syrian 12th grade student.\nStudent: {question}\nTutor:", max_length=120)
    return response[0]['generated_text'].split('Tutor:')[-1].strip()

print("✅ Emergency tutor ready!")
print("Test:", ask_tutor("Help me with English"))
```

This will give you a basic working tutor even if training failed!