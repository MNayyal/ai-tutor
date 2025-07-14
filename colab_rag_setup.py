"""
📚 CURRICULUM RAG PIPELINE FOR GOOGLE COLAB
===========================================

Simple setup for building an AI tutor that answers questions from your curriculum.

INSTRUCTIONS:
1. Run this cell to install dependencies and set up the system
2. Upload your curriculum markdown file when prompted
3. Ask questions about your curriculum!
"""

# ============================================================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================================================

print("🔧 Installing dependencies...")

# Uncomment these lines when running in Colab:
"""
!pip install langchain langchain-community
!pip install chromadb
!pip install sentence-transformers
!pip install transformers torch
!pip install gradio
"""

# ============================================================================
# STEP 2: IMPORT LIBRARIES
# ============================================================================

print("📚 Importing libraries...")

import os
import re
import warnings
warnings.filterwarnings('ignore')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import gradio as gr

print("✅ Libraries imported successfully!")

# ============================================================================
# STEP 3: UPLOAD AND PROCESS CURRICULUM
# ============================================================================

def upload_and_process_curriculum():
    """Upload and process the curriculum file"""
    
    # Upload file in Colab
    try:
        from google.colab import files
        print("📁 Please upload your curriculum markdown file:")
        uploaded = files.upload()
        file_name = list(uploaded.keys())[0]
        print(f"✅ File uploaded: {file_name}")
    except ImportError:
        # For local testing
        file_name = "english_textbook.md"
        print(f"⚠️  Using local file: {file_name}")
    
    # Read and process the file
    print("🔄 Processing curriculum content...")
    
    with open(file_name, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split into documents by units and sections
    documents = []
    
    # Find all units
    unit_pattern = r'# [Uu]nit (\d+)'
    unit_matches = list(re.finditer(unit_pattern, content))
    
    for i, match in enumerate(unit_matches):
        unit_num = match.group(1)
        start_pos = match.start()
        end_pos = unit_matches[i + 1].start() if i + 1 < len(unit_matches) else len(content)
        
        unit_content = content[start_pos:end_pos]
        
        # Split unit into sections
        sections = re.split(r'## ', unit_content)
        
        for section in sections[1:]:  # Skip first empty split
            section_title = section.split('\n')[0]
            
            doc = Document(
                page_content=f"Unit {unit_num} - {section_title}\n\n{section}",
                metadata={'unit': f'Unit {unit_num}', 'section': section_title}
            )
            documents.append(doc)
    
    print(f"✅ Created {len(documents)} curriculum sections")
    return documents

# ============================================================================
# STEP 4: CREATE VECTOR DATABASE
# ============================================================================

def create_vector_database(documents):
    """Create embeddings and vector database"""
    
    print("🧠 Creating embeddings...")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"📄 Created {len(split_docs)} text chunks")
    
    # Create vector database
    print("🗃️  Building vector database...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name="curriculum_db"
    )
    
    print("✅ Vector database created!")
    return vectorstore

# ============================================================================
# STEP 5: SETUP LANGUAGE MODEL
# ============================================================================

def setup_language_model():
    """Setup a free Hugging Face language model"""
    
    print("🤖 Loading language model...")
    
    # Use a lightweight model that works well in Colab
    model_name = "distilgpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=100,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)
    print("✅ Language model loaded!")
    return llm

# ============================================================================
# STEP 6: CREATE RAG SYSTEM
# ============================================================================

def create_rag_system(vectorstore, llm):
    """Create the complete RAG system"""
    
    print("🔗 Creating RAG system...")
    
    # Custom prompt for educational content
    prompt_template = """
You are an AI tutor helping students with their English curriculum. 
Use the following content to answer the question clearly and educationally.

Curriculum Content:
{context}

Question: {question}

Provide a helpful answer based on the curriculum content:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    print("✅ RAG system ready!")
    return qa_chain

# ============================================================================
# STEP 7: MAIN SETUP FUNCTION
# ============================================================================

def setup_curriculum_rag():
    """Main setup function - run this to initialize everything"""
    
    print("🚀 SETTING UP CURRICULUM RAG SYSTEM")
    print("=" * 50)
    
    # Step 1: Upload and process curriculum
    documents = upload_and_process_curriculum()
    
    # Step 2: Create vector database
    vectorstore = create_vector_database(documents)
    
    # Step 3: Setup language model
    llm = setup_language_model()
    
    # Step 4: Create RAG system
    qa_chain = create_rag_system(vectorstore, llm)
    
    print("\n🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("You can now ask questions about your curriculum!")
    
    return qa_chain, vectorstore

# ============================================================================
# STEP 8: QUESTION ANSWERING FUNCTIONS
# ============================================================================

def ask_question(qa_chain, question):
    """Ask a question and get an answer with sources"""
    
    try:
        result = qa_chain({"query": question})
        
        print(f"\n❓ Question: {question}")
        print(f"\n💡 Answer: {result['result']}")
        
        print(f"\n📚 Sources:")
        for i, doc in enumerate(result['source_documents'], 1):
            unit = doc.metadata.get('unit', 'Unknown')
            section = doc.metadata.get('section', 'Unknown')
            print(f"   {i}. {unit} - {section}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def search_curriculum(vectorstore, keyword, k=5):
    """Search for specific topics in the curriculum"""
    
    docs = vectorstore.similarity_search(keyword, k=k)
    
    print(f"\n🔍 Search results for '{keyword}':")
    for i, doc in enumerate(docs, 1):
        unit = doc.metadata.get('unit', 'Unknown')
        section = doc.metadata.get('section', 'Unknown')
        print(f"\n{i}. {unit} - {section}")
        print(f"   Preview: {doc.page_content[:150]}...")

# ============================================================================
# STEP 9: WEB INTERFACE
# ============================================================================

def create_web_interface(qa_chain):
    """Create a Gradio web interface"""
    
    def gradio_qa(question):
        if not question.strip():
            return "Please enter a question about your curriculum."
        
        try:
            result = qa_chain({"query": question})
            answer = result['result']
            
            # Add source information
            sources = "\n\n📚 Sources:\n"
            for doc in result['source_documents']:
                unit = doc.metadata.get('unit', 'Unknown')
                section = doc.metadata.get('section', 'Unknown')
                sources += f"• {unit} - {section}\n"
            
            return answer + sources
            
        except Exception as e:
            return f"Sorry, there was an error: {str(e)}"
    
    # Create interface
    interface = gr.Interface(
        fn=gradio_qa,
        inputs=gr.Textbox(
            label="Ask about your curriculum",
            placeholder="e.g., What factors affect career choice?",
            lines=2
        ),
        outputs=gr.Textbox(
            label="AI Tutor Response",
            lines=8
        ),
        title="📚 Curriculum AI Tutor",
        description="Ask questions about your English curriculum",
        examples=[
            ["What does success mean in the curriculum?"],
            ["How do parents influence their children's career choices?"],
            ["What vocabulary words are taught in Unit 1?"],
            ["What grammar topics are covered?"]
        ]
    )
    
    print("🌐 Launching web interface...")
    return interface.launch(share=True)

# ============================================================================
# STEP 10: QUICK START
# ============================================================================

def quick_start():
    """Quick start function - run this for immediate setup"""
    
    # Setup the RAG system
    qa_chain, vectorstore = setup_curriculum_rag()
    
    # Test with a sample question
    print("\n🧪 Testing with sample question...")
    ask_question(qa_chain, "What factors should be considered in career planning?")
    
    # Create web interface
    create_web_interface(qa_chain)
    
    # Return objects for further use
    return qa_chain, vectorstore

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================

usage_instructions = """
🎯 HOW TO USE:

1. Run this entire cell to install and setup everything
2. When prompted, upload your curriculum markdown file
3. Use these commands:

   # Quick setup (recommended)
   qa_chain, vectorstore = quick_start()
   
   # Ask individual questions
   ask_question(qa_chain, "Your question here")
   
   # Search for topics
   search_curriculum(vectorstore, "keyword")
   
   # Launch web interface
   create_web_interface(qa_chain)

🌟 EXAMPLE QUESTIONS:
- "What does success mean according to the curriculum?"
- "How do parents influence their children's career choices?"
- "What are the highlighted vocabulary words in Unit 1?"
- "What grammar exercises are included?"
- "What writing assignments are given?"

📱 The web interface will give you a shareable link to use the AI tutor!
"""

print(usage_instructions)

# ============================================================================
# AUTO-RUN FOR COLAB
# ============================================================================

# Uncomment the line below to automatically start setup when cell runs:
# qa_chain, vectorstore = quick_start()