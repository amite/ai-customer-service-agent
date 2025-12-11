"""
Customer Service Agent with LangChain Tool Calling
===================================================
This module demonstrates LangChain's tool calling capabilities using a customer 
service scenario. The agent can look up orders, check product inventory, handle 
refund requests, and perform semantic search using vector embeddings.
"""

import json
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import vector store functionality
from src.vectorstore.vector_store import initialize_vector_store, search_similar_products, search_faqs

# =============================================================================
# SAMPLE DATA: In-memory customer service database
# =============================================================================

# Sample customer orders with various statuses
ORDERS = [
    {
        "order_id": "ORD-1001",
        "customer_email": "john.doe@email.com",
        "status": "delivered",
        "order_date": "2024-11-15",
        "delivery_date": "2024-11-20",
        "items": [
            {"product_id": "PROD-A1", "name": "Wireless Headphones", "quantity": 1, "price": 89.99},
            {"product_id": "PROD-B2", "name": "USB-C Cable", "quantity": 2, "price": 12.99}
        ],
        "total": 115.97
    },
    {
        "order_id": "ORD-1002",
        "customer_email": "jane.smith@email.com",
        "status": "shipped",
        "order_date": "2024-12-01",
        "delivery_date": "2024-12-08",
        "items": [
            {"product_id": "PROD-C3", "name": "Laptop Stand", "quantity": 1, "price": 45.00}
        ],
        "total": 45.00
    },
    {
        "order_id": "ORD-1003",
        "customer_email": "john.doe@email.com",
        "status": "processing",
        "order_date": "2024-12-05",
        "delivery_date": None,
        "items": [
            {"product_id": "PROD-D4", "name": "Mechanical Keyboard", "quantity": 1, "price": 129.99}
        ],
        "total": 129.99
    },
    {
        "order_id": "ORD-1004",
        "customer_email": "bob.wilson@email.com",
        "status": "delivered",
        "order_date": "2024-11-10",
        "delivery_date": "2024-11-18",
        "items": [
            {"product_id": "PROD-E5", "name": "Webcam HD", "quantity": 1, "price": 79.99},
            {"product_id": "PROD-B2", "name": "USB-C Cable", "quantity": 1, "price": 12.99}
        ],
        "total": 92.98
    }
]

# Sample product inventory
INVENTORY = [
    {"product_id": "PROD-A1", "name": "Wireless Headphones", "stock": 45, "price": 89.99},
    {"product_id": "PROD-B2", "name": "USB-C Cable", "stock": 120, "price": 12.99},
    {"product_id": "PROD-C3", "name": "Laptop Stand", "stock": 8, "price": 45.00},
    {"product_id": "PROD-D4", "name": "Mechanical Keyboard", "stock": 0, "price": 129.99},
    {"product_id": "PROD-E5", "name": "Webcam HD", "stock": 23, "price": 79.99},
]

# Refund tracking (simulated database)
REFUNDS = []

# Global vector store (initialized once)
VECTORSTORE = None


# =============================================================================
# LANGCHAIN TOOLS: These are the functions the agent can call
# =============================================================================

@tool
def lookup_order(order_id: str, customer_email: Optional[str] = None) -> str:
    """
    Look up order details by order ID. Optionally verify with customer email.
    
    Args:
        order_id: The order ID to look up (e.g., "ORD-1001")
        customer_email: Optional customer email for verification
        
    Returns:
        JSON string with order details or error message
    """
    # Search for the order in our database
    for order in ORDERS:
        if order["order_id"] == order_id:
            # If email provided, verify it matches
            if customer_email and order["customer_email"] != customer_email:
                return json.dumps({
                    "error": "Order found but email does not match our records"
                })
            
            # Return order details
            return json.dumps({
                "success": True,
                "order": order
            })
    
    # Order not found
    return json.dumps({
        "error": f"Order {order_id} not found in our system"
    })


@tool
def check_product_availability(product_name: str) -> str:
    """
    Check if a product is in stock and get pricing information.
    
    Args:
        product_name: Name of the product to check (can be partial match)
        
    Returns:
        JSON string with product availability and details
    """
    # Search inventory for matching products (case-insensitive partial match)
    matching_products = [
        p for p in INVENTORY 
        if product_name.lower() in p["name"].lower()
    ]
    
    if not matching_products:
        return json.dumps({
            "error": f"No products found matching '{product_name}'"
        })
    
    # Format product information
    results = []
    for product in matching_products:
        results.append({
            "product_id": product["product_id"],
            "name": product["name"],
            "price": product["price"],
            "in_stock": product["stock"] > 0,
            "stock_count": product["stock"],
            "availability": "Available" if product["stock"] > 0 else "Out of Stock"
        })
    
    return json.dumps({
        "success": True,
        "products": results
    })


@tool
def search_products_semantic(query: str) -> str:
    """
    Search for products using semantic similarity (AI-powered search).
    This understands the MEANING of your query, not just keywords.
    
    Examples:
    - "something for gaming" → finds gaming keyboards
    - "improve my video calls" → finds webcams
    - "wireless audio" → finds headphones
    
    Args:
        query: Natural language description of what the customer wants
        
    Returns:
        JSON string with relevant product recommendations
    """
    global VECTORSTORE
    
    # Initialize vector store if not already done
    if VECTORSTORE is None:
        VECTORSTORE = initialize_vector_store()
    
    # Perform semantic search
    results = search_similar_products(VECTORSTORE, query, k=3)
    
    if not results:
        return json.dumps({
            "error": "No products found matching your description"
        })
    
    # Format for agent consumption
    formatted_products = []
    for result in results:
        formatted_products.append({
            "product_id": result["product_id"],
            "name": result["name"],
            "category": result["category"],
            "price": result["price"],
            "relevance": f"{(1 - result['similarity_score']) * 100:.1f}%",  # Convert distance to similarity %
            "description_excerpt": result["content"][:200] + "..."
        })
    
    return json.dumps({
        "success": True,
        "products": formatted_products,
        "search_type": "semantic"
    })


@tool
def search_knowledge_base(question: str) -> str:
    """
    Search the FAQ knowledge base for answers to common questions.
    Uses AI to understand the question and find relevant answers.
    
    Examples:
    - "How do I return something?" → return policy
    - "When will my order arrive?" → shipping information
    - "Do you have warranties?" → warranty details
    
    Args:
        question: Customer's question about policies, shipping, etc.
        
    Returns:
        JSON string with relevant FAQ answers
    """
    global VECTORSTORE
    
    # Initialize vector store if not already done
    if VECTORSTORE is None:
        VECTORSTORE = initialize_vector_store()
    
    # Search FAQs
    results = search_faqs(VECTORSTORE, question, k=2)
    
    if not results:
        return json.dumps({
            "error": "No relevant FAQ found for this question"
        })
    
    # Format results
    formatted_faqs = []
    for result in results:
        formatted_faqs.append({
            "question": result["question"],
            "answer": result["answer"],
            "category": result["category"],
            "relevance": f"{(1 - result['similarity_score']) * 100:.1f}%"
        })
    
    return json.dumps({
        "success": True,
        "faqs": formatted_faqs
    })


@tool
def process_refund_request(order_id: str, reason: str, customer_email: str) -> str:
    """
    Process a refund request for an order. Checks eligibility and creates refund ticket.
    
    Args:
        order_id: The order ID to refund
        reason: Reason for the refund request
        customer_email: Customer's email for verification
        
    Returns:
        JSON string with refund status and ticket information
    """
    # Look up the order
    order = None
    for o in ORDERS:
        if o["order_id"] == order_id:
            order = o
            break
    
    if not order:
        return json.dumps({
            "error": f"Order {order_id} not found"
        })
    
    # Verify customer email
    if order["customer_email"] != customer_email:
        return json.dumps({
            "error": "Email does not match order records"
        })
    
    # Check if order is eligible for refund (must be delivered and within 30 days)
    if order["status"] != "delivered":
        return json.dumps({
            "error": f"Order must be delivered before refund. Current status: {order['status']}"
        })
    
    # Check if delivery was within last 30 days
    delivery_date = datetime.strptime(order["delivery_date"], "%Y-%m-%d")
    days_since_delivery = (datetime.now() - delivery_date).days
    
    if days_since_delivery > 30:
        return json.dumps({
            "error": f"Refund window expired. Order delivered {days_since_delivery} days ago (limit: 30 days)"
        })
    
    # Create refund ticket
    refund_ticket = {
        "ticket_id": f"REF-{len(REFUNDS) + 1001}",
        "order_id": order_id,
        "customer_email": customer_email,
        "reason": reason,
        "amount": order["total"],
        "status": "pending",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "estimated_processing": "3-5 business days"
    }
    
    REFUNDS.append(refund_ticket)
    
    return json.dumps({
        "success": True,
        "message": "Refund request created successfully",
        "refund_ticket": refund_ticket
    })


# =============================================================================
# AGENT SETUP: Initialize LangChain components
# =============================================================================

def create_customer_service_agent():
    """
    Creates and configures the customer service agent with LangChain.
    Now includes vector search capabilities!
    
    Returns:
        AgentExecutor configured with tools and LLM
    """
    # Initialize Ollama LLM
    # Note: llama3.1 supports tool calling natively
    llm = ChatOllama(
        model="llama3.1:8b-instruct-q4_K_M",
        temperature=0,  # Use 0 for more deterministic responses
        num_predict=512,  # Max tokens to generate
    )
    
    # Compile list of available tools (now includes vector search!)
    tools = [
        lookup_order,
        check_product_availability,
        search_products_semantic,  # NEW: AI-powered product search
        search_knowledge_base,      # NEW: AI-powered FAQ search
        process_refund_request
    ]
    
    # Create the prompt template for the agent
    # This defines how the agent should behave and use tools
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful customer service agent for TechStore, an online electronics retailer.

Your role is to:
- Help customers track their orders
- Check product availability and pricing
- Recommend products based on customer needs (using semantic search)
- Answer policy questions using the knowledge base
- Process refund requests when eligible
- Provide friendly, professional assistance

Guidelines:
- Always verify customer email when handling sensitive operations (orders, refunds)
- Be empathetic and understanding
- Explain policies clearly (e.g., 30-day refund window)
- Use the available tools to get accurate, real-time information
- If you need information from the customer (like email or order ID), politely ask for it

IMPORTANT - Tool Selection:
- Use 'check_product_availability' for EXACT product name lookups (e.g., "do you have Wireless Headphones?")
- Use 'search_products_semantic' for DESCRIPTIVE queries (e.g., "I need something for gaming", "wireless audio equipment")
- Use 'search_knowledge_base' for policy/process questions (e.g., "how do returns work?", "what's your shipping policy?")
- Use 'lookup_order' for order status checks
- Use 'process_refund_request' for returns/refunds

Use the available tools to get accurate information when needed.
"""),
        # Placeholder for conversation history
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        # Current user message
        ("human", "{input}"),
        # Placeholder for agent's tool-calling scratchpad
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the tool-calling agent
    # This agent knows how to use tools based on the conversation
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Wrap in AgentExecutor to handle the tool-calling loop
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Print detailed execution steps
        handle_parsing_errors=True,  # Gracefully handle errors
        max_iterations=5,  # Prevent infinite loops
    )
    
    return agent_executor


# =============================================================================
# MAIN EXECUTION: For testing the agent directly
# =============================================================================

if __name__ == "__main__":
    """
    Standalone testing mode - run some example queries
    """
    print("🤖 Initializing Customer Service Agent with Vector Search...")
    agent = create_customer_service_agent()
    
    # Example queries to test (including semantic search!)
    test_queries = [
        "Can you look up order ORD-1001 for john.doe@email.com?",
        "I need something for gaming",  # Tests semantic search
        "What's your return policy?",   # Tests knowledge base
        "Do you have Wireless Headphones in stock?",  # Tests exact match
    ]
    
    print("\n" + "="*70)
    print("TESTING AGENT WITH SAMPLE QUERIES")
    print("="*70 + "\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {query}")
        print(f"{'='*70}\n")
        
        response = agent.invoke({
            "input": query,
            "chat_history": []
        })
        
        print(f"\n✓ RESPONSE:\n{response['output']}\n")
    
    print("\n" + "="*70)
    print("Testing complete! Run streamlit_app.py for interactive demo.")
    print("="*70)