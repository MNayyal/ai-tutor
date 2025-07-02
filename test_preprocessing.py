#!/usr/bin/env python3
"""
Test script to demonstrate data preprocessing for Syrian English Tutor
"""

import json
from data_preprocessor import TextbookPreprocessor

def main():
    print("🧪 Testing Data Preprocessing for Syrian English Tutor")
    print("=" * 60)
    
    # Check if textbook file exists
    textbook_file = "english_textbook.md"
    
    try:
        processor = TextbookPreprocessor(textbook_file)
        print(f"✅ Successfully loaded textbook: {textbook_file}")
        
        # Process the textbook
        results = processor.process_textbook()
        
        print(f"\n📊 Processing Results:")
        print(f"   📚 Units processed: {results['units']}")
        print(f"   💬 Conversations generated: {results['conversations']}")
        print(f"   📝 Instructions created: {results['instructions']}")
        
        # Load and display sample data
        with open('combined_training_data.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"\n📋 Training Data Summary:")
        print(f"   Total conversations: {len(data['conversations'])}")
        print(f"   Total instructions: {len(data['instructions'])}")
        print(f"   Source: {data['metadata']['source']}")
        
        # Show sample conversation
        if data['conversations']:
            print(f"\n💬 Sample Conversation:")
            sample = data['conversations'][0]
            print(f"   ID: {sample['conversation_id']}")
            for msg in sample['messages']:
                role = msg['role'].title()
                content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"   {role}: {content}")
        
        # Show sample instruction
        if data['instructions']:
            print(f"\n📝 Sample Instruction:")
            sample = data['instructions'][0]
            print(f"   Instruction: {sample['instruction']}")
            print(f"   Input: {sample['input'][:50]}..." if sample['input'] else "   Input: (none)")
            output = sample['output'][:100] + "..." if len(sample['output']) > 100 else sample['output']
            print(f"   Output: {output}")
        
        # Extract some statistics
        conversation_lengths = [len(conv['messages']) for conv in data['conversations']]
        avg_conv_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
        
        print(f"\n📈 Data Statistics:")
        print(f"   Average conversation length: {avg_conv_length:.1f} messages")
        print(f"   Shortest conversation: {min(conversation_lengths) if conversation_lengths else 0} messages")
        print(f"   Longest conversation: {max(conversation_lengths) if conversation_lengths else 0} messages")
        
        # Count different types of content
        vocab_conversations = len([c for c in data['conversations'] if 'vocab' in c['conversation_id']])
        topic_conversations = len([c for c in data['conversations'] if 'topic' in c['conversation_id']])
        exercise_conversations = len([c for c in data['conversations'] if 'exercise' in c['conversation_id']])
        
        print(f"\n🏷️ Content Types:")
        print(f"   Vocabulary conversations: {vocab_conversations}")
        print(f"   Topic conversations: {topic_conversations}")
        print(f"   Exercise conversations: {exercise_conversations}")
        print(f"   General help: {len(data['conversations']) - vocab_conversations - topic_conversations - exercise_conversations}")
        
        print(f"\n✅ Preprocessing test completed successfully!")
        print(f"📁 Generated files:")
        print(f"   - conversation_data.json")
        print(f"   - instruction_data.json")
        print(f"   - combined_training_data.json")
        
        print(f"\n🚀 Ready for training! You can now:")
        print(f"   1. Use the generated data with train_syrian_tutor.py")
        print(f"   2. Upload to Google Colab for training")
        print(f"   3. Customize the data further if needed")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find {textbook_file}")
        print(f"💡 Please ensure your textbook file is named '{textbook_file}' and is in the current directory")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        print(f"💡 Check that your textbook file is properly formatted")

if __name__ == "__main__":
    main()