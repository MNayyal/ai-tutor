#!/usr/bin/env python3
"""
Simple test script for Syrian English Tutor data preprocessing
No external dependencies required
"""

import re
import json
import os

def extract_simple_training_data(textbook_file):
    """Extract training data from the textbook using simple regex patterns"""
    print(f"📖 Reading textbook: {textbook_file}")
    
    with open(textbook_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    training_examples = []
    
    # Extract questions ending with question marks
    print("🔍 Extracting questions...")
    question_patterns = [
        r'(\d+\.)\s*(.+?\?)',
        r'([A-Z]\.)\s*(.+?\?)',
    ]
    
    questions_found = 0
    for pattern in question_patterns:
        matches = re.findall(pattern, content, re.MULTILINE)
        for num, question in matches:
            if len(question.strip()) > 10:  # Filter very short questions
                example = {
                    "input": f"Student: {question.strip()}",
                    "output": f"Tutor: Let me help you with this question. I'll explain this step by step based on the Syrian 12th grade English curriculum.",
                    "type": "question_answer"
                }
                training_examples.append(example)
                questions_found += 1
    
    print(f"   Found {questions_found} questions")
    
    # Extract vocabulary words (words in bold **word**)
    print("📚 Extracting vocabulary...")
    vocab_pattern = r'\*\*(\w+)\*\*'
    vocab_words = re.findall(vocab_pattern, content)
    
    # Try to find definitions for vocabulary words
    vocab_found = 0
    for word in set(vocab_words):  # Remove duplicates
        if len(word) > 2:  # Filter very short words
            # Look for the word followed by some explanation
            definition_pattern = rf'\*\*{word}\*\*\s*(.+?)(?=\n|\*\*|$)'
            definition_match = re.search(definition_pattern, content)
            
            if definition_match:
                definition = definition_match.group(1).strip()
                if len(definition) > 10:  # Filter very short definitions
                    example = {
                        "input": f"Student: What does '{word}' mean?",
                        "output": f"Tutor: The word '{word}' means: {definition}. Would you like me to provide some example sentences?",
                        "type": "vocabulary"
                    }
                    training_examples.append(example)
                    vocab_found += 1
            else:
                # Add a generic vocabulary example
                example = {
                    "input": f"Student: What does '{word}' mean?",
                    "output": f"Tutor: '{word}' is an important word from your Syrian 12th grade English curriculum. Let me explain its meaning and provide examples.",
                    "type": "vocabulary"
                }
                training_examples.append(example)
                vocab_found += 1
    
    print(f"   Found {vocab_found} vocabulary words")
    
    # Add general help examples
    print("🎯 Adding general tutoring examples...")
    general_examples = [
        {
            "input": "Student: Can you help me with Unit 1?",
            "output": "Tutor: I'd be happy to help you with Unit 1! This unit covers career planning and future goals. Let me guide you through the key topics step by step.",
            "type": "general_help"
        },
        {
            "input": "Student: I need help with grammar exercises.",
            "output": "Tutor: Grammar can be challenging, but I'm here to help! Let's work through the grammar concepts from your Syrian curriculum step by step.",
            "type": "grammar_help"
        },
        {
            "input": "Student: How do I prepare for my English exam?",
            "output": "Tutor: Great question! Let me give you some effective study strategies for your English exam based on the Syrian 12th grade curriculum.",
            "type": "exam_prep"
        },
        {
            "input": "Student: I'm struggling with vocabulary.",
            "output": "Tutor: Vocabulary building is important! Let me help you learn and remember new words from your textbook effectively.",
            "type": "vocab_help"
        },
        {
            "input": "Student: Help me understand this reading passage.",
            "output": "Tutor: I'd be happy to help you understand the reading passage! Let's break it down paragraph by paragraph and discuss the key ideas.",
            "type": "reading_help"
        },
        {
            "input": "Student: How do I write a good essay?",
            "output": "Tutor: Writing a good essay involves planning, structure, and clear expression. Let me guide you through the process using examples from your curriculum.",
            "type": "writing_help"
        }
    ]
    
    training_examples.extend(general_examples)
    print(f"   Added {len(general_examples)} general examples")
    
    return training_examples

def analyze_training_data(examples):
    """Analyze the extracted training data"""
    print(f"\n📊 Training Data Analysis:")
    print(f"   Total examples: {len(examples)}")
    
    # Count by type
    type_counts = {}
    for example in examples:
        example_type = example.get('type', 'unknown')
        type_counts[example_type] = type_counts.get(example_type, 0) + 1
    
    print(f"   📋 By type:")
    for example_type, count in type_counts.items():
        print(f"      {example_type}: {count}")
    
    # Show examples
    print(f"\n💬 Sample Examples:")
    for i, example in enumerate(examples[:3]):
        print(f"   Example {i+1}:")
        print(f"      Input: {example['input']}")
        print(f"      Output: {example['output'][:100]}...")
        print(f"      Type: {example['type']}")
        print()

def convert_to_training_format(examples):
    """Convert examples to the format expected by the training script"""
    training_texts = []
    
    for example in examples:
        # Convert to the <|startoftext|> format
        text = f"<|startoftext|>{example['input']}\n{example['output']}<|endoftext|>"
        training_texts.append(text)
    
    return training_texts

def save_training_data(examples, output_file='simple_training_data.json'):
    """Save the training data to a JSON file"""
    training_texts = convert_to_training_format(examples)
    
    data = {
        "training_texts": training_texts,
        "examples": examples,
        "metadata": {
            "source": "Syrian 12th Grade English Curriculum",
            "total_examples": len(examples),
            "total_training_texts": len(training_texts)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Saved training data to: {output_file}")
    return output_file

def main():
    print("🎓 Syrian English Tutor - Simple Data Processing Test")
    print("=" * 60)
    
    textbook_file = "english_textbook.md"
    
    if not os.path.exists(textbook_file):
        print(f"❌ Error: Could not find {textbook_file}")
        print(f"💡 Please ensure your textbook file is in the current directory")
        return
    
    try:
        # Extract training data
        examples = extract_simple_training_data(textbook_file)
        
        # Analyze the data
        analyze_training_data(examples)
        
        # Save the data
        output_file = save_training_data(examples)
        
        print(f"\n✅ Data processing completed successfully!")
        print(f"📁 Generated file: {output_file}")
        
        # Show sample training text
        with open(output_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if data['training_texts']:
            print(f"\n📝 Sample Training Text:")
            print(data['training_texts'][0])
        
        print(f"\n🚀 Next steps:")
        print(f"   1. This data can be used with the Google Colab training script")
        print(f"   2. Upload this file along with your textbook to Colab")
        print(f"   3. Modify the training script to use this preprocessed data")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()