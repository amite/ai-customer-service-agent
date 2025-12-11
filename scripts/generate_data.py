#!/usr/bin/env python3
"""
Data generation script for creating synthetic data for the RAG system
"""
import json
from pathlib import Path


def generate_sample_documents(num_docs: int = 10):
    """Generate sample documents for testing"""
    documents = []
    
    for i in range(num_docs):
        doc = {
            "id": f"doc_{i}",
            "content": f"This is sample document {i} with some content.",
            "metadata": {
                "source": "generated",
                "index": i
            }
        }
        documents.append(doc)
    
    return documents


def main():
    """Main function"""
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)
    
    documents = generate_sample_documents()
    
    output_file = output_dir / "sample_documents.json"
    with open(output_file, "w") as f:
        json.dump(documents, f, indent=2)
    
    print(f"Generated {len(documents)} documents -> {output_file}")


if __name__ == "__main__":
    main()
