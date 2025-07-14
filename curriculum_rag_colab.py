"""
Curriculum RAG Pipeline for Google Colab
AI Assistant for Student Questions from English Textbook

This script creates a Retrieval-Augmented Generation (RAG) pipeline using:
- LangChain for orchestration
- Chroma DB for vector storage
- Hugging Face models (free) for embeddings and generation
- English curriculum data for knowledge base
"""

print("Starting Curriculum RAG Pipeline Setup...")
print("=" * 60)

# ============================================================================
# 1. INSTALL REQUIRED PACKAGES
# ============================================================================

print("\n📦 Installing required packages...")

# Install packages (uncomment when running in Colab)
"""
!pip install langchain langchain-community langchain-huggingface
!pip install chromadb
!pip install sentence-transformers
!pip install transformers torch
!pip install accelerate bitsandbytes
!pip install gradio
"""

# ============================================================================
# 2. IMPORT LIBRARIES
# ============================================================================

print("📚 Importing libraries...")

import os
import re
from typing import List
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# Hugging Face imports
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Other imports
import gradio as gr
import json
from datetime import datetime

print("✅ All libraries imported successfully!")

# ============================================================================
# 3. DATA LOADING UTILITIES
# ============================================================================

def upload_curriculum_file():
    """
    Upload curriculum file in Colab environment
    """
    try:
        from google.colab import files
        print("Please upload your curriculum file (combined.md or english_textbook.md):")
        uploaded = files.upload()
        curriculum_file = list(uploaded.keys())[0]
        print(f"✅ Uploaded file: {curriculum_file}")
        return curriculum_file
    except ImportError:
        print("⚠️  Not in Colab environment. Using local file...")
        # For local testing, use the existing file
        return "english_textbook.md"

def load_and_process_curriculum(file_path: str) -> List[Document]:
    """
    Load the curriculum markdown file and process it into documents
    """
    print(f"📖 Loading curriculum from: {file_path}")
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split content by units and sections for better organization
    documents = []
    
    # Split by units first
    unit_pattern = r'# [Uu]nit (\d+)'
    unit_matches = list(re.finditer(unit_pattern, content))
    
    for i, match in enumerate(unit_matches):
        unit_num = match.group(1)
        start_pos = match.start()
        end_pos = unit_matches[i + 1].start() if i + 1 < len(unit_matches) else len(content)
        
        unit_content = content[start_pos:end_pos]
        
        # Further split each unit into sections
        section_pattern = r'## ([^\n]+)'
        section_matches = list(re.finditer(section_pattern, unit_content))
        
        for j, section_match in enumerate(section_matches):
            section_title = section_match.group(1).strip()
            section_start = section_match.start()
            section_end = section_matches[j + 1].start() if j + 1 < len(section_matches) else len(unit_content)
            
            section_content = unit_content[section_start:section_end]
            
            # Create metadata for better retrieval
            metadata = {
                'unit': f'Unit {unit_num}',
                'section': section_title,
                'source': file_path,
                'content_type': 'curriculum'
            }
            
            # Create document
            doc = Document(
                page_content=f"Unit {unit_num} - {section_title}\n\n{section_content}",
                metadata=metadata
            )
            documents.append(doc)
    
    print(f"✅ Loaded {len(documents)} curriculum sections")
    return documents

# ============================================================================
# 4. EMBEDDINGS SETUP
# ============================================================================

def setup_embeddings():
    """
    Set up embeddings using a free Hugging Face model
    """
    print("🔧 Setting up embeddings model...")
    
    # Using a lightweight but effective sentence transformer model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f"✅ Embeddings model loaded on: {device}")
    
    # Test embeddings
    test_embedding = embeddings.embed_query("What is career planning?")
    print(f"📏 Embedding dimension: {len(test_embedding)}")
    
    return embeddings

# ============================================================================
# 5. VECTOR DATABASE SETUP
# ============================================================================

def create_vector_database(documents: List[Document], embeddings):
    """
    Create vector database with Chroma
    """
    print("🗃️  Creating vector database...")
    
    # Split documents into smaller chunks for better retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"📄 Created {len(split_docs)} document chunks")
    
    # Create the vector store
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="curriculum_collection",
        persist_directory="./chroma_db"  # This will save the database
    )
    
    print(f"✅ Vector database created with {len(split_docs)} documents")
    
    # Test retrieval
    print("🔍 Testing retrieval...")
    test_query = "What factors should be considered in career selection?"
    retrieved_docs = vectorstore.similarity_search(test_query, k=3)
    
    print(f"📋 Retrieved {len(retrieved_docs)} documents for test query")
    for i, doc in enumerate(retrieved_docs[:2]):  # Show first 2
        print(f"   {i+1}. {doc.metadata.get('unit', 'Unknown')} - {doc.metadata.get('section', 'Unknown')}")
    
    return vectorstore, split_docs

# ============================================================================
# 6. LANGUAGE MODEL SETUP
# ============================================================================

def setup_language_model():
    """
    Set up the language model using a free Hugging Face model
    """
    print("🤖 Loading language model...")
    
    # Try different models in order of preference
    models_to_try = [
        "microsoft/DialoGPT-medium",
        "facebook/blenderbot-400M-distill", 
        "distilgpt2"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"   Trying {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Create pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            print(f"✅ Language model {model_name} loaded successfully!")
            return llm
            
        except Exception as e:
            print(f"   ❌ Failed to load {model_name}: {str(e)[:100]}...")
            continue
    
    raise Exception("Failed to load any language model")

# ============================================================================
# 7. RAG CHAIN SETUP
# ============================================================================

def create_rag_chain(llm, vectorstore):
    """
    Create the RAG chain with custom prompt
    """
    print("🔗 Creating RAG chain...")
    
    # Create a custom prompt template for educational Q&A
    prompt_template = """
You are an AI tutor helping students with their English curriculum. Use the following curriculum content to answer the student's question.

Context from curriculum:
{context}

Student's Question: {question}

Instructions:
- Provide a clear, educational answer based on the curriculum content
- If the question relates to exercises, help guide the student to the answer rather than giving it directly
- Include references to specific units or sections when relevant
- Keep your answer concise but informative
- If the information isn't in the curriculum, say so politely

Answer:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create the retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    print("✅ RAG chain created successfully!")
    return qa_chain

# ============================================================================
# 8. MAIN RAG CLASS
# ============================================================================

class CurriculumRAG:
    """
    Main RAG system class
    """
    
    def __init__(self, curriculum_file=None):
        self.curriculum_file = curriculum_file
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.conversation_history = []
        
    def setup(self):
        """
        Set up the complete RAG pipeline
        """
        print("🚀 Setting up Curriculum RAG Pipeline...")
        
        # 1. Load curriculum data
        if not self.curriculum_file:
            self.curriculum_file = upload_curriculum_file()
        
        documents = load_and_process_curriculum(self.curriculum_file)
        
        # 2. Setup embeddings
        self.embeddings = setup_embeddings()
        
        # 3. Create vector database
        self.vectorstore, self.split_docs = create_vector_database(documents, self.embeddings)
        
        # 4. Setup language model
        llm = setup_language_model()
        
        # 5. Create RAG chain
        self.qa_chain = create_rag_chain(llm, self.vectorstore)
        
        print("\n🎉 RAG Pipeline Setup Complete!")
        print("=" * 60)
        
    def ask_question(self, question: str):
        """
        Ask a question to the RAG system and get an answer with sources
        """
        if not self.qa_chain:
            print("❌ RAG system not initialized. Run setup() first.")
            return None
            
        try:
            result = self.qa_chain({"query": question})
            
            # Format response
            response = {
                "question": question,
                "answer": result['result'],
                "sources": []
            }
            
            for doc in result['source_documents']:
                source_info = {
                    "unit": doc.metadata.get('unit', 'Unknown Unit'),
                    "section": doc.metadata.get('section', 'Unknown Section'),
                    "content_preview": doc.page_content[:150] + "..."
                }
                response["sources"].append(source_info)
            
            # Save to conversation history
            self.conversation_history.append(response)
            
            # Print formatted response
            print(f"\n📚 Question: {question}")
            print(f"\n🤖 Answer: {response['answer']}")
            print(f"\n📖 Sources:")
            for i, source in enumerate(response['sources'], 1):
                print(f"   {i}. {source['unit']} - {source['section']}")
            
            return response
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def search_topics(self, keyword: str, k: int = 5):
        """
        Search for specific topics in the curriculum
        """
        if not self.vectorstore:
            print("❌ Vector database not initialized.")
            return []
            
        relevant_docs = self.vectorstore.similarity_search(keyword, k=k)
        
        print(f"\n🔍 Search results for '{keyword}':")
        results = []
        
        for i, doc in enumerate(relevant_docs, 1):
            unit = doc.metadata.get('unit', 'Unknown Unit')
            section = doc.metadata.get('section', 'Unknown Section')
            
            result = {
                "unit": unit,
                "section": section,
                "content": doc.page_content[:200] + "..."
            }
            results.append(result)
            
            print(f"\n{i}. {unit} - {section}")
            print(f"   Content preview: {result['content']}")
        
        return results
    
    def get_stats(self):
        """
        Get statistics about the curriculum content
        """
        if not hasattr(self, 'split_docs'):
            print("❌ Documents not loaded.")
            return
            
        print("\n📊 Curriculum Statistics:")
        print(f"   • Total document chunks: {len(self.split_docs)}")
        
        # Count units
        units = set([doc.metadata.get('unit', 'Unknown') for doc in self.split_docs])
        print(f"   • Number of units: {len(units)}")
        
        # Count sections per unit
        unit_sections = {}
        for doc in self.split_docs:
            unit = doc.metadata.get('unit', 'Unknown')
            section = doc.metadata.get('section', 'Unknown')
            if unit not in unit_sections:
                unit_sections[unit] = set()
            unit_sections[unit].add(section)
        
        print(f"   • Sections per unit:")
        for unit, sections in sorted(unit_sections.items()):
            print(f"     - {unit}: {len(sections)} sections")
    
    def create_gradio_interface(self):
        """
        Create Gradio web interface
        """
        if not self.qa_chain:
            print("❌ RAG system not initialized. Run setup() first.")
            return None
        
        def gradio_interface(question):
            if not question.strip():
                return "Please enter a question about your curriculum."
            
            response = self.ask_question(question)
            if not response:
                return "Sorry, I encountered an error processing your question."
            
            # Format the response for Gradio
            answer = response['answer']
            sources = "\n\n📚 **Sources from your curriculum:**\n"
            for source in response['sources']:
                sources += f"• {source['unit']} - {source['section']}\n"
            
            return answer + sources
        
        # Create Gradio interface
        interface = gr.Interface(
            fn=gradio_interface,
            inputs=[
                gr.Textbox(
                    label="Ask a question about your curriculum",
                    placeholder="e.g., What factors should be considered in career selection?",
                    lines=2
                )
            ],
            outputs=[
                gr.Textbox(
                    label="AI Tutor Response",
                    lines=10
                )
            ],
            title="📚 Curriculum AI Tutor",
            description="Ask questions about your English curriculum and get AI-powered answers based on your textbook content.",
            examples=[
                ["What does success mean according to the curriculum?"],
                ["How do parents influence career choices?"],
                ["What are the meanings of the highlighted words in Unit 1?"],
                ["What grammar topics are covered in Unit 1?"],
                ["What writing exercises are included in the curriculum?"]
            ],
            theme="huggingface"
        )
        
        print("🌐 Launching Gradio interface...")
        return interface.launch(share=True, debug=True)

# ============================================================================
# 9. USAGE EXAMPLES AND TESTING
# ============================================================================

def main():
    """
    Main function to demonstrate the RAG system
    """
    print("🎓 Curriculum RAG System Demo")
    print("=" * 40)
    
    # Initialize RAG system
    rag = CurriculumRAG(curriculum_file="english_textbook.md")  # Use local file for demo
    
    # Setup the pipeline
    rag.setup()
    
    # Show statistics
    rag.get_stats()
    
    # Test with sample questions
    sample_questions = [
        "What factors should be considered when choosing a career?",
        "What does the word 'aspire' mean according to the curriculum?",
        "How do parents influence their children's career choices?",
        "What are the components of success according to Unit 2?"
    ]
    
    print("\n🧪 Testing with sample questions...")
    print("=" * 40)
    
    for question in sample_questions:
        rag.ask_question(question)
        print("\n" + "-" * 60 + "\n")
    
    # Search for topics
    print("🔍 Topic search examples...")
    rag.search_topics("grammar")
    rag.search_topics("vocabulary")
    
    # Create web interface
    print("\n🌐 Creating web interface...")
    rag.create_gradio_interface()

# ============================================================================
# 10. QUICK START FOR COLAB
# ============================================================================

def quick_start():
    """
    Quick start function for immediate use in Colab
    """
    print("🚀 Quick Start: Curriculum RAG System")
    print("=" * 50)
    
    # Create RAG instance
    rag = CurriculumRAG()
    
    # Setup everything
    rag.setup()
    
    # Return the instance for further use
    return rag

# ============================================================================
# 11. COLAB INSTRUCTIONS
# ============================================================================

colab_instructions = """
🔥 GOOGLE COLAB INSTRUCTIONS:

1. Copy this entire script to a new Colab notebook cell
2. Uncomment the pip install commands at the top
3. Run the cell to install dependencies and set up the system
4. Upload your curriculum file when prompted
5. Use the following commands:

   # Quick setup
   rag = quick_start()
   
   # Ask questions
   rag.ask_question("Your question here")
   
   # Search topics
   rag.search_topics("keyword")
   
   # Launch web interface
   rag.create_gradio_interface()

📝 Example Questions:
- "What does success mean according to the curriculum?"
- "How do parents influence career choices?"
- "What are the meanings of highlighted words in Unit 1?"
- "What grammar topics are covered?"

🎯 Features:
- ✅ Free Hugging Face models
- ✅ Persistent vector database
- ✅ Interactive web interface
- ✅ Source citations
- ✅ Topic search
- ✅ Conversation history
"""

if __name__ == "__main__":
    print(colab_instructions)
    print("\n" + "=" * 60)
    main()