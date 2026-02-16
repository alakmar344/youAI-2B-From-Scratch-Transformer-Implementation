"""
Data Preparation Script for YouAI Training
Collects and prepares text data for training
"""

import os
import requests
import gzip
import json
from tqdm import tqdm
import random


class DataPreparer:
    """Prepare training data from various sources"""
    
    def __init__(self, output_dir='./data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def download_openwebtext(self):
        """
        Download OpenWebText dataset (40GB uncompressed)
        This is a recreation of OpenAI's WebText dataset
        """
        print("OpenWebText is a large dataset. Consider using:")
        print("1. Hugging Face datasets library")
        print("2. Manual download from: https://skylion007.github.io/OpenWebTextCorpus/")
        print("\nUsing Hugging Face:")
        print("from datasets import load_dataset")
        print("dataset = load_dataset('openwebtext')")
    
    def download_wikipedia(self, language='en'):
        """Download Wikipedia dumps"""
        print(f"To download Wikipedia dumps for {language}:")
        print(f"Visit: https://dumps.wikimedia.org/{language}wiki/latest/")
        print(f"Look for: {language}wiki-latest-pages-articles.xml.bz2")
        print("\nOr use:")
        print("from datasets import load_dataset")
        print(f"dataset = load_dataset('wikipedia', '20220301.{language}')")
    
    def download_gutenberg(self):
        """Download books from Project Gutenberg"""
        print("Project Gutenberg offers ~70,000 free books")
        print("Visit: https://www.gutenberg.org/")
        print("\nOr use the gutenberg library:")
        print("pip install gutenberg")
        print("from gutenberg.acquire import load_etext")
    
    def download_pile_sample(self, output_file='pile_sample.jsonl'):
        """
        Download a sample of The Pile dataset
        The Pile is an 800GB diverse dataset for language modeling
        """
        print("The Pile is a massive dataset. For a sample:")
        print("from datasets import load_dataset")
        print("dataset = load_dataset('EleutherAI/pile', split='train', streaming=True)")
        print("# Take first N examples")
        print("sample = dataset.take(100000)")
    
    def create_sample_dataset(self, num_examples=10000):
        """
        Create a small sample dataset for testing
        Uses random sentences for quick testing
        """
        print(f"Creating sample dataset with {num_examples} examples...")
        
        templates = [
            "The importance of {topic} cannot be overstated in modern {field}.",
            "Recent advances in {topic} have revolutionized the way we think about {concept}.",
            "Scientists have discovered that {topic} plays a crucial role in {process}.",
            "The relationship between {topic} and {concept} has been studied extensively.",
            "Understanding {topic} is essential for anyone working in {field}.",
            "Experts agree that {topic} will continue to shape the future of {field}.",
            "{topic} has emerged as a key factor in understanding {concept}.",
            "The impact of {topic} on {field} cannot be ignored by researchers.",
            "New research suggests that {topic} may be more important than previously thought.",
            "Many scholars have devoted their careers to studying {topic} and its effects.",
        ]
        
        topics = ["artificial intelligence", "climate change", "quantum computing", "biotechnology",
                 "renewable energy", "space exploration", "nanotechnology", "genetics", "robotics",
                 "neuroscience", "cryptography", "materials science", "ecology", "psychology"]
        
        fields = ["science", "technology", "medicine", "education", "industry", "research",
                 "engineering", "business", "society", "environment"]
        
        concepts = ["innovation", "sustainability", "efficiency", "development", "progress",
                   "understanding", "discovery", "implementation", "evolution", "transformation"]
        
        processes = ["learning", "adaptation", "growth", "change", "communication",
                    "interaction", "development", "evolution", "transformation", "optimization"]
        
        train_file = os.path.join(self.output_dir, 'train_data.txt')
        val_file = os.path.join(self.output_dir, 'val_data.txt')
        
        # Generate training data
        train_examples = []
        for _ in tqdm(range(num_examples), desc="Generating training data"):
            template = random.choice(templates)
            sentence = template.format(
                topic=random.choice(topics),
                field=random.choice(fields),
                concept=random.choice(concepts),
                process=random.choice(processes)
            )
            train_examples.append(sentence)
        
        # Generate validation data (10% of training size)
        val_examples = []
        for _ in tqdm(range(num_examples // 10), desc="Generating validation data"):
            template = random.choice(templates)
            sentence = template.format(
                topic=random.choice(topics),
                field=random.choice(fields),
                concept=random.choice(concepts),
                process=random.choice(processes)
            )
            val_examples.append(sentence)
        
        # Save to files
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(example + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(example + '\n')
        
        print(f"\n✅ Sample dataset created!")
        print(f"Training examples: {len(train_examples)}")
        print(f"Validation examples: {len(val_examples)}")
        print(f"Saved to: {self.output_dir}")
        
        return train_file, val_file
    
    def prepare_custom_text(self, text_files, train_split=0.9):
        """
        Prepare training data from custom text files
        
        Args:
            text_files: List of paths to text files
            train_split: Proportion of data for training
        """
        print("Preparing custom text data...")
        
        all_lines = []
        for file_path in text_files:
            print(f"Reading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                all_lines.extend([line.strip() for line in lines if line.strip()])
        
        print(f"Total lines: {len(all_lines)}")
        
        # Shuffle
        random.shuffle(all_lines)
        
        # Split
        split_idx = int(len(all_lines) * train_split)
        train_lines = all_lines[:split_idx]
        val_lines = all_lines[split_idx:]
        
        # Save
        train_file = os.path.join(self.output_dir, 'train_data.txt')
        val_file = os.path.join(self.output_dir, 'val_data.txt')
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for line in train_lines:
                f.write(line + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for line in val_lines:
                f.write(line + '\n')
        
        print(f"\n✅ Data prepared!")
        print(f"Training examples: {len(train_lines)}")
        print(f"Validation examples: {len(val_lines)}")
        
        return train_file, val_file
    
    def get_dataset_info(self):
        """Print information about popular datasets"""
        info = """
        ═══════════════════════════════════════════════════════════
        Popular Datasets for Training Language Models
        ═══════════════════════════════════════════════════════════
        
        1. THE PILE (825 GB)
           - Diverse dataset from EleutherAI
           - Books, websites, code, academic papers
           - Usage: load_dataset('EleutherAI/pile')
        
        2. OpenWebText (40 GB)
           - Recreation of OpenAI's WebText
           - Reddit-curated web pages
           - Usage: load_dataset('openwebtext')
        
        3. C4 (Colossal Clean Crawled Corpus) (305 GB)
           - Clean Common Crawl data
           - Usage: load_dataset('c4', 'en')
        
        4. Wikipedia (20 GB)
           - All of Wikipedia
           - Usage: load_dataset('wikipedia', '20220301.en')
        
        5. BookCorpus (5 GB)
           - 11,000 unpublished books
           - Usage: load_dataset('bookcorpus')
        
        6. CC-100 (2.5 TB)
           - Common Crawl in 100+ languages
           - Multilingual training
        
        7. mC4 (10+ TB)
           - Multilingual C4
           - 100+ languages
        
        ═══════════════════════════════════════════════════════════
        Recommendation for 2B model:
        - Minimum: 50-100 GB of clean text
        - Ideal: 200-500 GB of diverse text
        - For testing: Use sample dataset (this script)
        ═══════════════════════════════════════════════════════════
        """
        print(info)


def main():
    """Main function"""
    preparer = DataPreparer(output_dir='./data')
    
    print("=" * 60)
    print("YouAI Data Preparation")
    print("=" * 60 + "\n")
    
    print("Choose an option:")
    print("1. Create sample dataset (for testing - quick)")
    print("2. View information about real datasets")
    print("3. Prepare custom text files")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        num_examples = int(input("Number of training examples (default 10000): ") or "10000")
        preparer.create_sample_dataset(num_examples)
        
    elif choice == '2':
        preparer.get_dataset_info()
        print("\nTo download real datasets, use the Hugging Face datasets library:")
        print("\nfrom datasets import load_dataset")
        print("dataset = load_dataset('openwebtext')  # or any other dataset")
        print("\n# Convert to text file")
        print("with open('train_data.txt', 'w') as f:")
        print("    for example in dataset['train']:")
        print("        f.write(example['text'] + '\\n')")
        
    elif choice == '3':
        files = input("Enter text file paths (comma-separated): ").strip().split(',')
        files = [f.strip() for f in files]
        preparer.prepare_custom_text(files)
    
    else:
        print("Invalid choice!")
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("You can now run: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
