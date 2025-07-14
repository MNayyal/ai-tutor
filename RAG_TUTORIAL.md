# 📚 Curriculum RAG Pipeline Tutorial

## AI-Powered Student Assistant for Curriculum Questions

This project creates a **Retrieval-Augmented Generation (RAG)** system that helps students get answers from their English curriculum textbook using AI. The system uses **free Hugging Face models** and is optimized for **Google Colab**.

---

## 🎯 What This Does

✅ **Answers student questions** based on curriculum content  
✅ **Provides source citations** from specific units and sections  
✅ **Web interface** for easy interaction  
✅ **Topic search** functionality  
✅ **Free to use** - no API keys required  
✅ **Runs in Google Colab** - no local setup needed  

---

## 🚀 Quick Start (Google Colab)

### Option 1: Simple Setup (Recommended)

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Create a new notebook** and copy the entire contents of `colab_rag_setup.py`

3. **Uncomment the installation lines** at the top:
   ```python
   !pip install langchain langchain-community
   !pip install chromadb
   !pip install sentence-transformers
   !pip install transformers torch
   !pip install gradio
   ```

4. **Run the cell** - it will install dependencies and setup the system

5. **Upload your curriculum file** when prompted (your `combined.md` file)

6. **Start using the AI tutor**:
   ```python
   # Quick setup
   qa_chain, vectorstore = quick_start()
   ```

### Option 2: Advanced Setup

1. Upload the comprehensive `curriculum_rag_colab.py` file to Colab
2. Follow the detailed instructions in the notebook
3. Customize the system as needed

---

## 💡 How to Use

### Basic Commands

```python
# Ask a question
ask_question(qa_chain, "What factors should be considered in career planning?")

# Search for topics
search_curriculum(vectorstore, "grammar")

# Launch web interface
create_web_interface(qa_chain)
```

### Example Questions You Can Ask

- **Vocabulary**: "What does the word 'aspire' mean according to the curriculum?"
- **Content**: "How do parents influence their children's career choices?"
- **Grammar**: "What grammar topics are covered in Unit 1?"
- **Exercises**: "What writing assignments are given in the curriculum?"
- **Concepts**: "What does success mean according to Unit 2?"

---

## 🛠️ Technical Architecture

### Components Used

1. **LangChain**: Orchestrates the RAG pipeline
2. **Chroma DB**: Vector database for storing embeddings
3. **Sentence Transformers**: Creates text embeddings (`all-MiniLM-L6-v2`)
4. **Hugging Face Transformers**: Language model for generation (`distilgpt2`)
5. **Gradio**: Web interface for interaction

### How It Works

```
📖 Curriculum Text → 🔄 Text Chunks → 🧠 Embeddings → 🗃️ Vector DB
                                                          ↓
📱 Web Interface ← 🤖 AI Response ← 🔗 RAG Chain ← 🔍 Retrieval
```

1. **Document Processing**: Splits curriculum into manageable chunks
2. **Embedding Creation**: Converts text to numerical vectors
3. **Vector Storage**: Stores embeddings in Chroma database
4. **Question Processing**: Converts user question to embedding
5. **Retrieval**: Finds relevant curriculum sections
6. **Generation**: AI creates answer based on retrieved content
7. **Response**: Returns answer with source citations

---

## 📁 Project Files

- **`colab_rag_setup.py`**: Simple, ready-to-use Colab script
- **`curriculum_rag_colab.py`**: Comprehensive RAG pipeline with advanced features
- **`rag_requirements.txt`**: Python dependencies
- **`english_textbook.md`**: Your curriculum data (128KB, 2365 lines)
- **`RAG_TUTORIAL.md`**: This tutorial file

---

## 🔧 Customization Options

### Modify the Language Model

```python
# Try different models (in setup_language_model function)
model_options = [
    "microsoft/DialoGPT-medium",      # Better for conversations
    "facebook/blenderbot-400M-distill", # Good for Q&A
    "distilgpt2",                     # Lightweight, fast
    "google/flan-t5-base"             # Good for instructions
]
```

### Adjust Retrieval Settings

```python
# Change number of retrieved documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Default is 3

# Modify chunk size
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,    # Larger chunks = more context
    chunk_overlap=100  # More overlap = better continuity
)
```

### Custom Prompts

```python
# Modify the prompt template for different behavior
prompt_template = """
You are a strict tutor. Answer only based on the curriculum content.
If information is not in the curriculum, say "This topic is not covered."

Curriculum Content: {context}
Question: {question}

Answer:
"""
```

---

## 🎨 Web Interface Features

The Gradio interface provides:

- **Question Input**: Text box for asking questions
- **AI Response**: Formatted answer with sources
- **Example Questions**: Pre-written examples to try
- **Shareable Link**: Public URL to share with others
- **Real-time Processing**: Instant responses

### Interface Screenshot Description
```
┌─────────────────────────────────────┐
│     📚 Curriculum AI Tutor         │
├─────────────────────────────────────┤
│ Ask about your curriculum:          │
│ [What factors affect career choice?]│
│                            [Submit] │
├─────────────────────────────────────┤
│ AI Tutor Response:                  │
│ According to the curriculum...      │
│                                     │
│ 📚 Sources:                         │
│ • Unit 1 - Career Planning         │
│ • Unit 1 - Introductory Questions  │
└─────────────────────────────────────┘
```

---

## 🚨 Troubleshooting

### Common Issues and Solutions

**1. Installation Errors**
```python
# If you get dependency conflicts, try:
!pip install --upgrade pip
!pip install --no-deps langchain
```

**2. Memory Issues in Colab**
```python
# Use lighter models:
model_name = "distilgpt2"  # Instead of larger models
chunk_size = 300           # Smaller chunks
```

**3. Upload Issues**
```python
# If file upload fails, try:
from google.colab import files
import time
time.sleep(2)  # Wait before uploading
uploaded = files.upload()
```

**4. Model Loading Errors**
```python
# Clear GPU memory:
import torch
torch.cuda.empty_cache()
```

---

## 📊 Performance Tips

### For Better Results

1. **Curriculum Preparation**:
   - Ensure your markdown file is well-formatted
   - Use clear headings (`# Unit 1`, `## Section`)
   - Remove excessive formatting that might confuse the parser

2. **Question Formulation**:
   - Be specific: "What does 'aspire' mean?" vs "Define aspire"
   - Reference units: "What grammar is taught in Unit 1?"
   - Ask for examples: "Give examples of phrasal verbs from the curriculum"

3. **System Optimization**:
   - Use GPU runtime in Colab for faster processing
   - Restart runtime if memory issues occur
   - Cache the vector database to avoid rebuilding

---

## 🔮 Advanced Features

### Conversation History

```python
# Access conversation history
for conversation in rag.conversation_history:
    print(f"Q: {conversation['question']}")
    print(f"A: {conversation['answer'][:100]}...")
```

### Topic Analysis

```python
# Get curriculum statistics
rag.get_stats()

# Search specific topics
results = rag.search_topics("vocabulary", k=10)
```

### Export Functionality

```python
# Save conversation history
import json
with open('conversation_log.json', 'w') as f:
    json.dump(rag.conversation_history, f, indent=2)
```

---

## 🎓 Educational Use Cases

### For Students
- **Homework Help**: Get explanations of curriculum concepts
- **Study Review**: Test understanding with questions
- **Vocabulary**: Look up word meanings from context
- **Exam Prep**: Practice with curriculum-based questions

### For Teachers
- **Content Verification**: Check if information is in curriculum
- **Question Generation**: Get ideas for test questions
- **Curriculum Mapping**: Find where topics are covered
- **Student Support**: Provide 24/7 AI tutor access

### For Parents
- **Homework Assistance**: Help children with schoolwork
- **Progress Tracking**: Understand what children are learning
- **Supplementary Learning**: Extra practice outside class

---

## 🔒 Privacy and Security

- **Local Processing**: All data stays in your Colab session
- **No API Keys**: Uses free, open-source models
- **Temporary Storage**: Data deleted when session ends
- **No Data Collection**: Your curriculum content isn't stored externally

---

## 🌟 Next Steps

### Possible Enhancements

1. **Multi-Language Support**: Translate questions and answers
2. **Image Processing**: Handle diagrams and figures
3. **Quiz Generation**: Automatically create practice tests
4. **Progress Tracking**: Monitor student learning over time
5. **Integration**: Connect with learning management systems

### Contributing

This is an open framework - feel free to:
- Add new curriculum subjects
- Improve the AI prompts
- Enhance the web interface
- Add new features

---

## 📞 Support

If you encounter issues:

1. **Check the troubleshooting section** above
2. **Restart your Colab runtime** and try again
3. **Verify your curriculum file format** matches the expected structure
4. **Try the simple setup** if the advanced version fails

---

## 🎉 Success Indicators

You'll know the system is working when:

✅ File uploads successfully  
✅ Vector database creation completes  
✅ Language model loads without errors  
✅ Test question returns relevant answer  
✅ Web interface launches with shareable link  
✅ Questions return answers with source citations  

---

**Happy Learning! 🚀📚**

*Your AI-powered curriculum assistant is ready to help students succeed!*