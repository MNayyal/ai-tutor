# 🎓 Syrian English Tutor AI

Train a personalized AI tutor for Syrian 12th grade English curriculum using Google Colab and Hugging Face models.

## 🌟 Overview

This project creates an AI tutor specifically designed for Syrian 12th grade English students. The tutor can help with:

- 📚 **Vocabulary explanations** - Define words and provide examples
- 📝 **Grammar assistance** - Explain grammar rules and help with exercises  
- 📖 **Reading comprehension** - Help understand texts and answer questions
- ✍️ **Writing guidance** - Assist with essays and compositions
- 🎯 **Exam preparation** - Create practice questions and study guides
- 🌍 **Cultural context** - Understand content in Syrian educational context

## 🚀 Quick Start (Google Colab)

### Option 1: Step-by-Step Colab (Recommended)

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Create a new notebook** and enable GPU:
   - Click "Runtime" → "Change runtime type" → Select "GPU" → Save

3. **Copy the code**: Use the content from `colab_quick_start.py` - copy each section into separate Colab cells

4. **Run each cell** in order:
   - Cell 1: Install dependencies
   - Cell 2: Upload your `english_textbook.md` file
   - Cell 3: Preprocess the data
   - Cell 4-7: Setup and configure the model
   - Cell 8: Train the model (20-40 minutes)
   - Cell 9: Test the model
   - Cell 10: Launch interactive interface
   - Cell 11: Download trained model

### Option 2: One-Click Training

1. **Upload files** to Colab:
   - `train_syrian_tutor.py`
   - `data_preprocessor.py`
   - `english_textbook.md`

2. **Run the training script**:
   ```python
   !python train_syrian_tutor.py
   ```

## 📋 Requirements

### For Google Colab (Free Tier)
- Google account
- Your textbook file (`english_textbook.md`)
- ~1-2 hours for complete training

### For Local Training
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- Dependencies from `requirements.txt`

## 📁 File Structure

```
syrian-english-tutor/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── data_preprocessor.py          # Advanced data preprocessing
├── train_syrian_tutor.py         # Complete training script
├── colab_quick_start.py          # Step-by-step Colab version
├── english_textbook.md           # Your textbook (you provide)
└── deployment/                   # Generated after training
    ├── model/                    # Trained model files
    ├── tutor.py                 # Simple usage script
    └── README.md                # Deployment instructions
```

## 🔧 Technical Details

### Model Architecture
- **Base Model**: Microsoft DialoGPT-small (117M parameters)
- **Training Method**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Training Data**: Extracted Q&A pairs and vocabulary from textbook
- **Context**: Optimized for Syrian 12th grade English curriculum

### Memory Optimization
- Uses LoRA to reduce trainable parameters by 95%
- Gradient checkpointing to reduce memory usage
- Small batch sizes compatible with free Colab
- Mixed precision training (FP16) when GPU available

### Training Configuration
- **Epochs**: 2-3 (adjustable)
- **Batch Size**: 1 per device, accumulated to 8
- **Learning Rate**: 5e-5
- **Max Sequence Length**: 256 tokens
- **Training Time**: 20-40 minutes on free Colab

## 📚 Data Processing

The system automatically extracts training data from your textbook:

1. **Questions & Answers**: Finds numbered questions and creates Q&A pairs
2. **Vocabulary**: Extracts bold words and their definitions  
3. **General Help**: Adds common tutoring scenarios
4. **Conversational Format**: Formats as Student-Tutor dialogues

Example training data:
```
<|startoftext|>
Student: What does 'aspire' mean?
Tutor: The word 'aspire' means: to seek to attain a goal. Would you like me to provide some example sentences?
<|endoftext|>
```

## 🎨 Interface Features

The Gradio interface provides:

- **Chat Interface**: Natural conversation with the AI tutor
- **Example Buttons**: Quick access to common questions
- **Clear Chat**: Reset conversation history
- **Responsive Design**: Works on desktop and mobile
- **Public Sharing**: Shareable link for students

## 💡 Usage Examples

### Vocabulary Help
- **Student**: "What does 'burden' mean?"
- **Tutor**: "The word 'burden' means: load. It refers to something heavy that you carry, or responsibilities that feel heavy..."

### Grammar Assistance  
- **Student**: "I need help with present perfect tense"
- **Tutor**: "I'd be happy to help you with present perfect tense! This tense is formed with 'have/has' + past participle..."

### Exam Preparation
- **Student**: "How should I prepare for my English exam?"
- **Tutor**: "Great question! Let me give you some effective study strategies for your English exam..."

## 🔧 Customization

### Adjusting Training Parameters

In the training script, you can modify:

```python
# Training duration
num_train_epochs=3  # Increase for more training

# Model size
MODEL_NAME = "microsoft/DialoGPT-medium"  # Use larger model

# Sequence length
max_length=512  # Longer sequences (needs more memory)

# Batch size
per_device_train_batch_size=2  # Larger batches (needs more memory)
```

### Adding Custom Training Data

You can enhance the training by adding your own examples:

```python
custom_examples = [
    ("How do I improve my English?", "Here are some effective ways to improve your English skills..."),
    ("Explain passive voice", "Passive voice is used when the focus is on the action, not who does it...")
]
```

## 🚨 Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce `per_device_train_batch_size` to 1
   - Reduce `max_length` to 128
   - Use DialoGPT-small instead of medium

2. **Training Too Slow**
   - Ensure GPU is enabled in Colab
   - Check if CUDA is available: `torch.cuda.is_available()`
   - Reduce number of training examples

3. **Poor Model Performance**
   - Increase training epochs
   - Add more diverse training data
   - Use a larger base model

4. **Interface Not Loading**
   - Check if model files exist in the output directory
   - Ensure all dependencies are installed
   - Try restarting the runtime

### Memory Requirements

| Configuration | GPU Memory | Training Time |
|---------------|------------|---------------|
| Minimal       | 4GB        | 20 min        |
| Recommended   | 8GB        | 30 min        |
| Optimal       | 16GB       | 15 min        |

## 📈 Performance Optimization

### For Better Results
1. **More Training Data**: Add additional textbooks or materials
2. **Longer Training**: Increase epochs to 5-10
3. **Larger Model**: Use DialoGPT-medium or GPT-2
4. **Fine-tuned Prompts**: Customize the conversation format

### For Faster Training
1. **Smaller Model**: Use DistilGPT-2
2. **Reduced Sequence Length**: Set max_length=128
3. **Fewer Examples**: Filter training data for quality
4. **Higher Learning Rate**: Try 1e-4 for faster convergence

## 🌍 Deployment Options

### 1. Gradio Public Link
- Automatic sharing via Gradio
- Free hosting for 72 hours
- Accessible via web browser

### 2. Local Deployment
- Download the trained model
- Run `python tutor.py` locally
- Full control over hosting

### 3. Cloud Deployment
- Deploy to Hugging Face Spaces
- Use platforms like Streamlit Cloud
- Host on AWS/GCP/Azure

## 📝 License & Ethics

### Educational Use
This project is designed for educational purposes to help Syrian students learn English. Please ensure:

- **Respectful Use**: Use the AI tutor as a learning aid, not a replacement for human teachers
- **Privacy**: Don't share personal information with the AI
- **Accuracy**: Always verify important information with textbooks or teachers
- **Fair Use**: Use the textbook content in accordance with educational fair use

### Responsible AI
- The model may occasionally provide incorrect information
- Encourage critical thinking and verification
- Use as a supplement to, not replacement for, traditional learning
- Monitor usage, especially with younger students

## 🤝 Contributing

We welcome contributions to improve the Syrian English Tutor:

1. **Report Issues**: Submit bugs or suggestions via GitHub issues
2. **Improve Data Processing**: Enhance the textbook parsing algorithms  
3. **Add Features**: Contribute new functionality to the interface
4. **Documentation**: Help improve this guide

## 📞 Support

Need help? Here are your options:

1. **Check the Troubleshooting section** above
2. **Review the example outputs** to ensure expected behavior
3. **Start with the simple Colab version** before trying advanced features
4. **Verify your textbook file** is properly formatted markdown

## 🎯 Future Improvements

Planned enhancements:
- **Multi-language support** (Arabic explanations)
- **Voice interaction** for pronunciation help
- **Progress tracking** for students
- **Teacher dashboard** for monitoring
- **Mobile app** version
- **Integration** with learning management systems

## 🙏 Acknowledgments

This project uses:
- [Hugging Face Transformers](https://huggingface.co/transformers/) for the AI models
- [Gradio](https://gradio.app/) for the web interface  
- [LoRA](https://arxiv.org/abs/2106.09685) for efficient fine-tuning
- [Google Colab](https://colab.research.google.com/) for free GPU access

Special thanks to the Syrian education community for providing curriculum context and feedback.

---

## 🎓 Ready to Start?

1. **Gather your materials**: Ensure you have your `english_textbook.md` file ready
2. **Open Google Colab**: Navigate to [colab.research.google.com](https://colab.research.google.com)
3. **Follow the quick start guide** above
4. **Train your tutor**: Run the training process (20-40 minutes)
5. **Share with students**: Use the generated interface to help students learn

**Happy teaching! 🚀📚**

Your AI tutor will help Syrian students excel in English while providing personalized, culturally-aware educational support.

---

*Last updated: December 2024*
