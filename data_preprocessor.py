import re
import json
from typing import List, Dict, Tuple
import markdown
from bs4 import BeautifulSoup

class TextbookPreprocessor:
    def __init__(self, markdown_file_path: str):
        self.markdown_file_path = markdown_file_path
        self.conversation_data = []
        self.qa_pairs = []
        
    def extract_content(self) -> str:
        """Read and extract content from markdown file"""
        with open(self.markdown_file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    
    def parse_units(self, content: str) -> List[Dict]:
        """Parse the content into structured units"""
        units = []
        
        # Split by major units
        unit_pattern = r'# Unit \d+|# Module \d+'
        unit_splits = re.split(unit_pattern, content)
        
        for i, unit_content in enumerate(unit_splits[1:], 1):  # Skip first empty split
            unit_data = {
                'unit_number': i,
                'content': unit_content.strip(),
                'topics': self.extract_topics(unit_content),
                'vocabulary': self.extract_vocabulary(unit_content),
                'grammar': self.extract_grammar(unit_content),
                'exercises': self.extract_exercises(unit_content)
            }
            units.append(unit_data)
        
        return units
    
    def extract_topics(self, content: str) -> List[str]:
        """Extract main topics from unit content"""
        topics = []
        
        # Find headings that indicate topics
        topic_patterns = [
            r'## (.+?)(?=\n|$)',
            r'### (.+?)(?=\n|$)'
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, content)
            topics.extend([match.strip() for match in matches])
        
        return topics
    
    def extract_vocabulary(self, content: str) -> List[Dict]:
        """Extract vocabulary items and their definitions"""
        vocab_items = []
        
        # Pattern for vocabulary definitions (word - definition pairs)
        vocab_patterns = [
            r'(\w+)\s*:\s*(.+?)(?=\n|$)',
            r'(\*\*\w+\*\*)\s*(.+?)(?=\n|$)',
            r'(\d+\.\s*\*\*\w+\*\*)\s*(.+?)(?=\n|$)'
        ]
        
        for pattern in vocab_patterns:
            matches = re.findall(pattern, content)
            for word, definition in matches:
                vocab_items.append({
                    'word': word.strip('*'),
                    'definition': definition.strip()
                })
        
        return vocab_items
    
    def extract_grammar(self, content: str) -> List[str]:
        """Extract grammar rules and explanations"""
        grammar_sections = []
        
        # Find grammar sections
        grammar_pattern = r'## Grammar(.+?)(?=## |$)'
        matches = re.findall(grammar_pattern, content, re.DOTALL)
        
        for match in matches:
            grammar_sections.append(match.strip())
        
        return grammar_sections
    
    def extract_exercises(self, content: str) -> List[Dict]:
        """Extract exercises and questions"""
        exercises = []
        
        # Pattern for questions (numbered items with question marks)
        question_patterns = [
            r'(\d+\.)\s*(.+?\?)',
            r'([A-Z]\.)\s*(.+?\?)',
            r'(\*\*Questions:\*\*|Questions:)(.+?)(?=\*\*|##|$)'
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for num, question in matches:
                exercises.append({
                    'number': num.strip(),
                    'question': question.strip(),
                    'type': 'comprehension'
                })
        
        return exercises
    
    def create_training_conversations(self, units: List[Dict]) -> List[Dict]:
        """Create conversation-style training data"""
        conversations = []
        
        for unit in units:
            unit_num = unit['unit_number']
            
            # Create conversations for different types of interactions
            
            # 1. Topic explanation conversations
            for topic in unit['topics']:
                conversation = {
                    "conversation_id": f"unit_{unit_num}_topic_{len(conversations)}",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"Can you help me understand the topic '{topic}' from Unit {unit_num}?"
                        },
                        {
                            "role": "assistant", 
                            "content": f"I'd be happy to help you with '{topic}' from Unit {unit_num}. This topic is part of the Syrian 12th grade English curriculum. Let me explain the key concepts and provide examples to help you understand better."
                        }
                    ]
                }
                conversations.append(conversation)
            
            # 2. Vocabulary help conversations
            for vocab in unit['vocabulary']:
                conversation = {
                    "conversation_id": f"unit_{unit_num}_vocab_{len(conversations)}",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"What does '{vocab['word']}' mean?"
                        },
                        {
                            "role": "assistant",
                            "content": f"The word '{vocab['word']}' means: {vocab['definition']}. Would you like me to provide some example sentences or help you practice using this word in context?"
                        }
                    ]
                }
                conversations.append(conversation)
            
            # 3. Exercise help conversations
            for exercise in unit['exercises']:
                conversation = {
                    "conversation_id": f"unit_{unit_num}_exercise_{len(conversations)}",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"I need help with this question: {exercise['question']}"
                        },
                        {
                            "role": "assistant",
                            "content": f"Let me help you with this question. First, let's break down what the question is asking. Then I'll guide you through the solution step by step."
                        }
                    ]
                }
                conversations.append(conversation)
            
            # 4. General study help conversations
            general_conversations = [
                {
                    "conversation_id": f"unit_{unit_num}_general_help",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"I'm studying Unit {unit_num}. Can you help me prepare for my exam?"
                        },
                        {
                            "role": "assistant",
                            "content": f"Absolutely! I can help you prepare for your Unit {unit_num} exam. Let's review the key topics, practice vocabulary, work through grammar exercises, and I can create practice questions for you. What specific area would you like to focus on first?"
                        }
                    ]
                },
                {
                    "conversation_id": f"unit_{unit_num}_writing_help",
                    "messages": [
                        {
                            "role": "user",
                            "content": f"I need help with the writing assignment in Unit {unit_num}."
                        },
                        {
                            "role": "assistant",
                            "content": f"I'd be happy to help you with your writing assignment! Let's break down the requirements, create an outline, and I'll guide you through structuring your composition. I can also help with grammar, vocabulary choice, and proofreading."
                        }
                    ]
                }
            ]
            conversations.extend(general_conversations)
        
        return conversations
    
    def create_instruction_dataset(self, units: List[Dict]) -> List[Dict]:
        """Create instruction-following dataset"""
        instructions = []
        
        for unit in units:
            unit_num = unit['unit_number']
            
            # Create instruction examples for different tasks
            task_instructions = [
                {
                    "instruction": f"Explain the main concepts from Unit {unit_num} of the Syrian 12th grade English curriculum.",
                    "input": "",
                    "output": f"Unit {unit_num} covers several important topics for 12th grade English students in Syria. Let me break down the key concepts..."
                },
                {
                    "instruction": "Help a student understand difficult vocabulary words.",
                    "input": f"I don't understand these words from Unit {unit_num}: {', '.join([v['word'] for v in unit['vocabulary'][:3]])}",
                    "output": "I'll help you understand these vocabulary words. Let me explain each one with its meaning and provide examples..."
                },
                {
                    "instruction": "Provide guidance for English grammar exercises.",
                    "input": "I'm confused about the grammar exercises in this unit.",
                    "output": "Grammar can be challenging, but I'm here to help! Let me guide you through the grammar concepts step by step..."
                },
                {
                    "instruction": "Act as a patient and encouraging English tutor.",
                    "input": "I'm struggling with English and feel discouraged.",
                    "output": "I understand that learning English can feel challenging sometimes, but you're making progress! Let's work together to identify areas where you need support and create a study plan that works for you."
                }
            ]
            
            instructions.extend(task_instructions)
        
        return instructions
    
    def save_training_data(self, conversations: List[Dict], instructions: List[Dict], output_dir: str = "."):
        """Save processed training data to files"""
        
        # Save conversation data
        with open(f"{output_dir}/conversation_data.json", 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        # Save instruction data
        with open(f"{output_dir}/instruction_data.json", 'w', encoding='utf-8') as f:
            json.dump(instructions, f, indent=2, ensure_ascii=False)
        
        # Create a combined dataset
        combined_data = {
            "conversations": conversations,
            "instructions": instructions,
            "metadata": {
                "source": "Syrian 12th Grade English Curriculum",
                "total_conversations": len(conversations),
                "total_instructions": len(instructions)
            }
        }
        
        with open(f"{output_dir}/combined_training_data.json", 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training data saved:")
        print(f"- Conversations: {len(conversations)}")
        print(f"- Instructions: {len(instructions)}")
        print(f"- Files saved in: {output_dir}")
    
    def process_textbook(self, output_dir: str = ".") -> Dict:
        """Main processing pipeline"""
        print("Starting textbook preprocessing...")
        
        # Extract content
        content = self.extract_content()
        print("✓ Content extracted")
        
        # Parse into units
        units = self.parse_units(content)
        print(f"✓ Parsed {len(units)} units")
        
        # Create training data
        conversations = self.create_training_conversations(units)
        instructions = self.create_instruction_dataset(units)
        print("✓ Training data created")
        
        # Save data
        self.save_training_data(conversations, instructions, output_dir)
        
        return {
            "units": len(units),
            "conversations": len(conversations),
            "instructions": len(instructions)
        }

# Example usage
if __name__ == "__main__":
    processor = TextbookPreprocessor("english_textbook.md")
    results = processor.process_textbook()
    print(f"Processing complete! Generated {results['conversations']} conversations and {results['instructions']} instructions.")