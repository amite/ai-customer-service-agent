"""
Vector Store Setup with Qdrant and Embeddings
==============================================
This module sets up a Qdrant vector store with product descriptions and FAQs.
Uses mxbai-embed-large for semantic search capabilities.
"""

import os
from typing import List, Dict
from langchain_qdrant import QdrantVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from qdrant_client.models import Distance, Filter, FieldCondition, MatchValue

# =============================================================================
# SAMPLE DATA: Product descriptions for semantic search
# =============================================================================

# Detailed product descriptions that will be embedded
PRODUCT_DESCRIPTIONS = [
    {
        "product_id": "PROD-A1",
        "name": "Wireless Headphones",
        "category": "Audio",
        "description": "Premium wireless Bluetooth headphones with active noise cancellation. "
                      "Features 30-hour battery life, comfortable over-ear design, and superior "
                      "sound quality. Perfect for music lovers, travelers, and remote workers. "
                      "Includes carrying case and USB-C charging cable.",
        "price": 89.99,
        "tags": ["bluetooth", "noise-cancelling", "wireless", "audio", "headphones"]
    },
    {
        "product_id": "PROD-B2",
        "name": "USB-C Cable",
        "category": "Accessories",
        "description": "Durable braided USB-C to USB-C cable supporting fast charging up to 100W "
                      "and data transfer speeds of 480Mbps. 6-foot length provides flexibility. "
                      "Compatible with laptops, tablets, smartphones, and other USB-C devices. "
                      "Reinforced connectors prevent fraying.",
        "price": 12.99,
        "tags": ["usb-c", "cable", "charging", "data-transfer", "accessories"]
    },
    {
        "product_id": "PROD-C3",
        "name": "Laptop Stand",
        "category": "Ergonomics",
        "description": "Adjustable aluminum laptop stand improves posture and reduces neck strain. "
                      "Compatible with laptops 10-17 inches. Features 6 height adjustments, "
                      "ventilated design for cooling, and non-slip silicone pads. Foldable for "
                      "easy portability. Ideal for home office and remote work setups.",
        "price": 45.00,
        "tags": ["ergonomic", "laptop", "stand", "adjustable", "office"]
    },
    {
        "product_id": "PROD-D4",
        "name": "Mechanical Keyboard",
        "category": "Input Devices",
        "description": "RGB backlit mechanical gaming keyboard with tactile switches. Features "
                      "customizable per-key lighting, programmable macros, and anti-ghosting "
                      "technology. Durable aluminum frame with detachable wrist rest. Perfect "
                      "for gamers, programmers, and typing enthusiasts. Available with blue, "
                      "brown, or red switches.",
        "price": 129.99,
        "tags": ["mechanical", "keyboard", "gaming", "rgb", "programmable"]
    },
    {
        "product_id": "PROD-E5",
        "name": "Webcam HD",
        "category": "Video",
        "description": "1080p HD webcam with built-in dual microphones and auto-focus. Features "
                      "low-light correction and 90-degree wide-angle lens. Ideal for video "
                      "conferencing, streaming, and content creation. Universal clip fits most "
                      "monitors and laptops. Plug-and-play USB connection, no drivers required.",
        "price": 79.99,
        "tags": ["webcam", "video", "hd", "conferencing", "streaming"]
    },
]

# FAQ data for semantic search
FAQ_DATA = [
    {
        "question": "What is your return policy?",
        "answer": "We offer a 30-day return policy for all products. Items must be in original "
                 "condition with packaging. Refunds are processed within 3-5 business days after "
                 "we receive the return. Shipping costs are non-refundable unless the item was "
                 "defective or we made an error.",
        "category": "Returns & Refunds"
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Expedited shipping (2-3 days) and "
                 "overnight shipping are available at checkout for additional fees. Once your "
                 "order ships, you'll receive a tracking number via email.",
        "category": "Shipping"
    },
    {
        "question": "Do you offer warranty on products?",
        "answer": "Yes! All products come with a 1-year manufacturer warranty covering defects "
                 "in materials and workmanship. Extended warranty plans (2-3 years) are available "
                 "for purchase at checkout. Warranty does not cover accidental damage or normal "
                 "wear and tear.",
        "category": "Warranty"
    },
    {
        "question": "Can I track my order?",
        "answer": "Absolutely! Once your order ships, we'll send you a tracking number via email. "
                 "You can also log into your account on our website to view real-time tracking "
                 "information. Orders typically ship within 1-2 business days of placement.",
        "category": "Order Tracking"
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept all major credit cards (Visa, Mastercard, American Express, Discover), "
                 "PayPal, Apple Pay, Google Pay, and Shop Pay. For orders over $500, we also offer "
                 "financing options through Affirm with 0% APR for qualified buyers.",
        "category": "Payment"
    },
    {
        "question": "Do you ship internationally?",
        "answer": "Currently, we ship to the United States, Canada, and select European countries. "
                 "International shipping costs are calculated at checkout based on destination and "
                 "package weight. Please note that customers are responsible for any customs fees "
                 "or import duties.",
        "category": "Shipping"
    },
    {
        "question": "How can I contact customer support?",
        "answer": "Our customer support team is available Monday-Friday, 9 AM - 6 PM EST. You can "
                 "reach us via email at support@techstore.com, live chat on our website, or phone "
                 "at 1-800-TECH-STORE. We typically respond to emails within 24 hours.",
        "category": "Support"
    },
    {
        "question": "Are your products genuine/authentic?",
        "answer": "Yes, we only sell 100% authentic products sourced directly from manufacturers or "
                 "authorized distributors. Every product comes with proper documentation and "
                 "manufacturer warranty. We never sell counterfeit or gray-market items.",
        "category": "Product Authenticity"
    },
]


# =============================================================================
# VECTOR STORE INITIALIZATION
# =============================================================================

def initialize_vector_store(collection_name: str = "techstore_products", use_persistent: bool = True):
    """
    Initialize Qdrant vector store with embeddings.
    
    This function:
    1. Creates an embedding model using Ollama's mxbai-embed-large
    2. Prepares documents from product and FAQ data
    3. Creates a Qdrant collection (persistent by default)
    4. Stores embedded documents in the collection
    
    Args:
        collection_name: Name for the Qdrant collection
        use_persistent: Whether to use persistent storage (default: True)
        
    Returns:
        QdrantVectorStore instance ready for semantic search
    """
    print("🔧 Initializing embeddings model...")
    
    # Initialize Ollama embeddings with mxbai-embed-large model
    # This model converts text into 1024-dimensional vectors
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large:latest",
    )
    
    print("📄 Preparing documents for embedding...")
    
    # Prepare product documents
    # Each document contains the searchable text and metadata
    product_docs = []
    for product in PRODUCT_DESCRIPTIONS:
        # Combine name and description for richer semantic search
        content = f"{product['name']}: {product['description']}"
        
        # Create Document with metadata
        doc = Document(
            page_content=content,
            metadata={
                "product_id": product["product_id"],
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "tags": ", ".join(product["tags"]),
                "type": "product"  # Distinguish from FAQs
            }
        )
        product_docs.append(doc)
    
    # Prepare FAQ documents
    faq_docs = []
    for faq in FAQ_DATA:
        # Combine question and answer for semantic search
        content = f"Q: {faq['question']}\nA: {faq['answer']}"
        
        doc = Document(
            page_content=content,
            metadata={
                "question": faq["question"],
                "answer": faq["answer"],
                "category": faq["category"],
                "type": "faq"  # Distinguish from products
            }
        )
        faq_docs.append(doc)
    
    # Combine all documents
    all_documents = product_docs + faq_docs
    
    print(f"📦 Embedding {len(all_documents)} documents...")
    
    # Determine storage location
    if use_persistent:
        # Use project root directory for persistent storage
        # This file is at: project_root/src/vectorstore/vector_store.py
        # So we go up 3 levels to reach project root
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        storage_path = os.path.join(project_root, "qdrant_storage")
        os.makedirs(storage_path, exist_ok=True)
        print(f"💾 Using persistent storage at: {storage_path}")
        
        # Create Qdrant vector store with path parameter for local persistent storage
        # The 'path' parameter enables on-disk storage instead of in-memory
        vectorstore = QdrantVectorStore.from_documents(
            documents=all_documents,
            embedding=embeddings,
            path=storage_path,  # Use 'path' parameter for local persistent storage
            collection_name=collection_name,
            distance=Distance.COSINE,
            force_recreate=True,  # Recreate collection each time for fresh data
        )
    else:
        # For in-memory storage, use location parameter
        print("⚠️  Using in-memory storage (metadata filtering may be limited)")
        vectorstore = QdrantVectorStore.from_documents(
            documents=all_documents,
            embedding=embeddings,
            location=":memory:",
            collection_name=collection_name,
            distance=Distance.COSINE,
            force_recreate=True,  # Recreate collection each time for fresh data
        )
    
    print(f"✅ Vector store initialized with {len(all_documents)} documents!")
    
    return vectorstore


def search_similar_products(vectorstore: QdrantVectorStore, query: str, k: int = 3) -> List[Dict]:
    """
    Search for products using semantic similarity.
    
    This performs a vector similarity search to find products that match
    the user's intent, even if they don't use exact keywords.
    
    Args:
        vectorstore: Initialized QdrantVectorStore
        query: User's search query (e.g., "something for gaming")
        k: Number of results to return
        
    Returns:
        List of dictionaries with product information and similarity scores
    """
    # Perform similarity search with Qdrant filter
    # This properly filters at the database level (works with persistent storage)
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
        filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.type",  # Note: use "metadata.type" for nested field
                    match=MatchValue(value="product")
                )
            ]
        )
    )
    
    # Format results
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "product_id": doc.metadata.get("product_id"),
            "name": doc.metadata.get("name"),
            "category": doc.metadata.get("category"),
            "price": doc.metadata.get("price"),
            "content": doc.page_content,
            "similarity_score": score,  # Lower score = more similar (cosine distance)
            "tags": doc.metadata.get("tags")
        })
    
    return formatted_results


def search_faqs(vectorstore: QdrantVectorStore, query: str, k: int = 2) -> List[Dict]:
    """
    Search for relevant FAQs using semantic similarity.
    
    This helps answer common questions by finding the most relevant FAQ
    based on the user's query meaning, not just keywords.
    
    Args:
        vectorstore: Initialized QdrantVectorStore
        query: User's question (e.g., "how do I send something back?")
        k: Number of results to return
        
    Returns:
        List of dictionaries with FAQ information
    """
    # Perform similarity search on FAQs only with proper Qdrant filter
    results = vectorstore.similarity_search_with_score(
        query=query,
        k=k,
        filter=Filter(
            must=[
                FieldCondition(
                    key="metadata.type",  # Note: use "metadata.type" for nested field
                    match=MatchValue(value="faq")
                )
            ]
        )
    )
    
    # Format results
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "question": doc.metadata.get("question"),
            "answer": doc.metadata.get("answer"),
            "category": doc.metadata.get("category"),
            "similarity_score": score,
        })
    
    return formatted_results


# =============================================================================
# DEMO/TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Standalone testing mode - demonstrates vector search capabilities
    """
    print("="*70)
    print("VECTOR STORE DEMO - WITH PERSISTENT STORAGE")
    print("="*70 + "\n")
    
    # Initialize vector store with persistent storage
    vectorstore = initialize_vector_store(use_persistent=True)
    
    # Test product searches
    print("\n" + "="*70)
    print("PRODUCT SEARCH TESTS")
    print("="*70 + "\n")
    
    test_queries = [
        "I need something for gaming",
        "wireless audio equipment",
        "ergonomic office setup",
        "video call equipment"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 70)
        results = search_similar_products(vectorstore, query, k=2)
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['name']} (${result['price']})")
            print(f"   Category: {result['category']}")
            print(f"   Score: {result['similarity_score']:.4f}")
            print(f"   Tags: {result['tags']}")
    
    # Test FAQ searches
    print("\n\n" + "="*70)
    print("FAQ SEARCH TESTS")
    print("="*70 + "\n")
    
    faq_queries = [
        "Can I send an item back if I don't like it?",
        "How long until my package arrives?",
        "Do your products have warranty?"
    ]
    
    for query in faq_queries:
        print(f"\n❓ Query: '{query}'")
        print("-" * 70)
        results = search_faqs(vectorstore, query, k=1)
        for result in results:
            print(f"\nQ: {result['question']}")
            print(f"A: {result['answer']}")
            print(f"Category: {result['category']}")
            print(f"Score: {result['similarity_score']:.4f}")
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70)