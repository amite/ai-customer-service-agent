# LangChain Tool Calling Project: Customer Service Agent 🤖

A complete, production-ready example demonstrating **LangChain tool calling** with **Ollama** and **Streamlit**. This project teaches you how to build AI agents that can call functions, manage state, and interact with structured data.

## 📚 What You'll Learn

- **Tool Calling Basics**: How LLMs can invoke Python functions dynamically
- **Agent Creation**: Building LangChain agents with custom tools
- **Structured Data**: Working with JSON data in AI applications  
- **State Management**: Maintaining conversation context across interactions
- **UI Integration**: Building interactive frontends with Streamlit
- **Error Handling**: Gracefully managing agent failures

---

## 🎯 Project Overview

### Business Domain: Customer Service

This agent simulates a customer service representative for "TechStore" that can:

1. **Look up orders** by order ID and verify customer email
2. **Check product availability** and pricing from inventory
3. **Search products semantically** using AI-powered vector search (understands intent, not just keywords!)
4. **Answer policy questions** using vector-based FAQ search
5. **Process refund requests** with eligibility validation

### Architecture

```
┌─────────────┐
│  Streamlit  │  ← User Interface
│     UI      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  LangChain  │  ← Agent orchestration
│    Agent    │     (AgentExecutor)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Ollama    │  ← LLM (Llama 3.1) - decides which tools to call
│   LLM API   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Python    │  ← Tools executed by agent
│    Tools    │     - lookup_order
└──────┬──────┘     - check_product_availability
       │            - process_refund_request
       │            - search_products_semantic ──┐
       │            - search_knowledge_base ─────┤
       │                                         │
       │                                         ▼
       │                              ┌──────────────────┐
       │                              │  OllamaEmbeddings│  ← Embedding model
       │                              │ (mxbai-embed-    │     (separate from LLM)
       │                              │  large)          │
       │                              └────────┬─────────┘
       │                                       │
       │                                       ▼
       │                              ┌──────────────────┐
       │                              │   Qdrant         │  ← Vector similarity search
       │                              │ Vector Store     │     (persistent storage)
       │                              └──────────────────┘
       │
       └──────────────────────────────────────┘
                    (results flow back to agent)
```

---

## 🚀 Setup Instructions

### Prerequisites

✅ **WSL (Windows Subsystem for Linux)** - Already installed  
✅ **Ollama** - Already installed with required models  
✅ **UV** - Python package manager  
✅ **Python 3.10+**

### Step 1: Verify Ollama Models

Ensure you have the required models downloaded:

```bash
# Check installed models
ollama list

# You should see:
# - llama3.1:8b-instruct-q4_K_M (for agent reasoning)
# - mxbai-embed-large:latest (for embeddings, if needed later)
```

### Step 2: Create Project Directory

```bash
# Create and navigate to project folder
mkdir langchain-customer-service
cd langchain-customer-service
```

### Step 3: Initialize UV Project

```bash
# Initialize a new UV project
uv init

# Add dependencies
uv add langchain langchain-ollama langchain-core streamlit
```

**Alternative**: Use the provided `pyproject.toml` file:

```bash
# If you have pyproject.toml from this tutorial
uv sync
```

### Step 4: Add Project Files

Create these files in your project directory:

1. `customer_service_agent.py` - Core agent logic with tools (see artifact above)
2. `vector_store.py` - Qdrant vector store and embedding setup (see artifact above)
3. `streamlit_app.py` - Web interface (see artifact above)
4. `pyproject.toml` - Dependencies (see artifact above)

### Step 5: Run the Application

#### Option A: Streamlit UI (Recommended)

```bash
# Activate the UV environment and run Streamlit
uv run streamlit run streamlit_app.py
```

This will:
- Start a local web server (usually at `http://localhost:8501`)
- Open your browser automatically
- Display an interactive chat interface

#### Option B: Command Line Testing

```bash
# Run the agent directly for testing
uv run python customer_service_agent.py
```

This runs predefined test queries and shows agent reasoning in the terminal.

---

## 🧠 Understanding the Code

### 1. Vector Embeddings & Semantic Search

**What are embeddings?**
Embeddings convert text into numerical vectors (arrays of numbers) that capture semantic meaning:

```python
# Text
"gaming keyboard" 

# Gets converted to embedding (1024 dimensions for mxbai-embed-large)
[0.234, -0.891, 0.456, ...] # 1024 numbers

# Similar text has similar vectors
"mechanical keyboard for gaming" 
[0.228, -0.887, 0.461, ...] # Close to above!
```

**How semantic search works:**

```python
# Initialize embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large:latest")

# Create vector store
vectorstore = QdrantVectorStore.from_documents(
    documents=product_docs,
    embedding=embeddings,
    location=":memory:",  # In-memory for demo
    collection_name="techstore_products"
)

# Search by meaning, not keywords!
results = vectorstore.similarity_search("something for gaming", k=3)
# Returns: Mechanical Keyboard, Webcam HD, etc.
```

**Why this is powerful:**
- User says "wireless audio" → finds "Wireless Headphones"
- User says "video conferencing" → finds "Webcam HD"
- User says "ergonomic work setup" → finds "Laptop Stand"
- No exact keyword match needed!

### 2. Tool Definition with `@tool` Decorator

LangChain tools are Python functions decorated with `@tool`. The LLM can call these based on their docstrings:

```python
from langchain_core.tools import tool

@tool
def search_products_semantic(query: str) -> str:
    """
    Search for products using semantic similarity (AI-powered search).
    This understands the MEANING of your query, not just keywords.
    
    Examples:
    - "something for gaming" → finds gaming keyboards
    - "improve my video calls" → finds webcams
    
    Args:
        query: Natural language description of what the customer wants
        
    Returns:
        JSON string with relevant product recommendations
    """
    # Vector search happens here
    results = search_similar_products(VECTORSTORE, query, k=3)
    return json.dumps({"success": True, "products": results})
```

**Key Points:**
- Docstring is crucial - the LLM reads it to understand when/how to use the tool
- Examples in docstring help the LLM decide when to use this tool
- Type hints help with structured data validation
- Return JSON strings for structured responses
- Tools should be deterministic and handle errors gracefully

### 2. Agent Creation Pattern

```python
def create_customer_service_agent():
    # 1. Initialize LLM
    llm = ChatOllama(
        model="llama3.1:8b-instruct-q4_K_M",
        temperature=0,  # Deterministic for consistent behavior
    )
    
    # 2. Define available tools (including vector search!)
    tools = [
        lookup_order, 
        check_product_availability,
        search_products_semantic,  # AI-powered product search
        search_knowledge_base,      # AI-powered FAQ search  
        process_refund_request
    ]
    
    # 3. Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful customer service agent..."),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # 4. Create tool-calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # 5. Wrap in executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=5,
    )
    
    return agent_executor
```

**Breakdown:**

- **LLM**: The "brain" that decides which tools to call
- **Tools**: Functions the agent can execute
- **Prompt**: Instructions for agent behavior
- **Agent**: Decision-making logic
- **AgentExecutor**: Orchestrates the tool-calling loop

### 3. The Tool-Calling Loop (with Vector Search!)

When you invoke the agent with a semantic query:

```python
response = agent.invoke({
    "input": "I need something for gaming",
    "chat_history": []
})
```

This happens behind the scenes:

```
1. User Input → Agent
2. Agent analyzes request → Recognizes descriptive query (not exact product name)
3. Agent decides to use "search_products_semantic" tool
4. Tool converts query to embedding vector using mxbai-embed-large
5. Qdrant finds most similar product vectors
6. Tool returns: [{"name": "Mechanical Keyboard", "relevance": "94.2%", ...}]
7. Agent receives results → Formulates natural response
8. Agent returns: "I found some great gaming products! Our Mechanical Keyboard 
   with RGB lighting and programmable macros would be perfect for gaming..."
```

**Traditional keyword search vs Semantic search:**

```python
# Traditional (keyword matching)
"gaming" in "Mechanical Keyboard"  # False - misses it!

# Semantic (meaning matching)
similarity("gaming", "Mechanical Keyboard with RGB lighting for gamers")
# High similarity score! - finds it!
```

### 4. Conversation Memory

```python
# Store previous messages
chat_history = [
    HumanMessage(content="What's my order status?"),
    AIMessage(content="I can help! What's your order ID?"),
    HumanMessage(content="ORD-1001"),
]

# Agent remembers context
response = agent.invoke({
    "input": "What about refunds?",
    "chat_history": chat_history
})
```

The agent now knows we're discussing order ORD-1001 and can reference it.

### 5. Structured Data Handling

Tools work with structured data (dictionaries) and return JSON:

```python
# Sample order data structure
order = {
    "order_id": "ORD-1001",
    "customer_email": "john.doe@email.com",
    "status": "delivered",
    "items": [
        {"product_id": "PROD-A1", "name": "Headphones", "price": 89.99}
    ],
    "total": 89.99
}

# Tool returns JSON string
return json.dumps({"success": True, "order": order})
```

### 6. Vector Store Configuration

Qdrant is used for efficient similarity search:

```python
# Create vector store with embeddings
vectorstore = QdrantVectorStore.from_documents(
    documents=all_documents,           # Product + FAQ documents
    embedding=embeddings,               # mxbai-embed-large model
    location=":memory:",                # In-memory (use path for persistence)
    collection_name="techstore_products",
    distance=Distance.COSINE,          # Similarity metric
)

# Documents are structured like this:
document = Document(
    page_content="Wireless Headphones: Premium wireless Bluetooth...",
    metadata={
        "product_id": "PROD-A1",
        "name": "Wireless Headphones",
        "category": "Audio",
        "price": 89.99,
        "type": "product"  # For filtering
    }
)
```

**Key concepts:**
- **Documents**: Text content + metadata
- **Embeddings**: Convert text → vectors (1024 dimensions)
- **Cosine Distance**: Measures vector similarity (0 = identical)
- **In-memory**: Fast but temporary (use file path for persistence)
- **Metadata filtering**: Search only products or only FAQs

---

## 💡 Example Queries to Try

### Semantic Search (AI-Powered! 🤖)
```
"I need something for gaming"
"Show me wireless audio equipment"
"What do you have for video calls?"
"I want to improve my ergonomic setup"
```

**What makes this special:**
- No exact keywords needed!
- Understands "gaming" → recommends mechanical keyboards
- Understands "video calls" → suggests webcams
- Uses vector embeddings to match meaning, not just text

### Basic Order Lookup
```
"Can you look up order ORD-1001 for john.doe@email.com?"
```

**What happens:**
1. Agent recognizes need for order lookup
2. Calls `lookup_order("ORD-1001", "john.doe@email.com")`
3. Returns formatted order details

### Product Availability Check
```
"Do you have Wireless Headphones in stock?"
```

**What happens:**
1. Agent identifies exact product query
2. Calls `check_product_availability("Wireless Headphones")`
3. Returns stock status and pricing

### Knowledge Base Search (Vector-Powered! 🧠)
```
"What's your return policy?"
"How long does shipping take?"
"Do you offer warranties?"
```

**What happens:**
1. Agent uses semantic search on FAQ database
2. Finds most relevant answers based on meaning
3. Returns policy information

### Refund Processing
```
"I'd like to return order ORD-1001. The product is defective."
```

**What happens:**
1. Agent recognizes refund request
2. Calls `lookup_order` first to verify eligibility
3. Then calls `process_refund_request` with details
4. Creates refund ticket and explains process

### Multi-Turn Conversation
```
User: "What's the status of my order?"
Agent: "I can help! What's your order ID?"
User: "ORD-1002"
Agent: "And your email for verification?"
User: "jane.smith@email.com"
Agent: [Looks up order and provides status]
```

**What happens:**
- Agent maintains context across messages
- Asks follow-up questions when needed
- Remembers previous user responses

---

## 🎓 Learning Exercises

### Exercise 1: Test Vector Search Directly

Run the vector store module to see embeddings in action:

```bash
uv run python vector_store.py
```

This will demonstrate:
- How "gaming" query finds mechanical keyboards
- How "wireless audio" finds headphones
- Similarity scores for each match

### Exercise 2: Add a New Tool

Create a tool to check order shipping estimates:

```python
@tool
def get_shipping_estimate(order_id: str) -> str:
    """
    Get estimated shipping time for an order.
    
    Args:
        order_id: The order ID to check
        
    Returns:
        JSON string with shipping estimate
    """
    # Your implementation here
    pass
```

**Hints:**
- Look up the order status
- Calculate days based on status (processing=5-7 days, shipped=2-3 days)
- Return JSON with estimate

### Exercise 3: Add Validation Logic

Enhance the refund tool to check if items are returnable:

```python
# Some items might not be eligible for return
NON_RETURNABLE_PRODUCTS = ["PROD-B2"]  # USB cables

# Add logic to process_refund_request to check this
```

### Exercise 4: Implement Email Notifications

Create a tool that simulates sending confirmation emails:

```python
@tool
def send_confirmation_email(email: str, subject: str, message: str) -> str:
    """Send a confirmation email to the customer."""
    # Log the email instead of actually sending
    pass
```

### Exercise 5: Expand Vector Store (Advanced)

Add more product data and categories:

```python
# In vector_store.py, add new products
NEW_PRODUCTS = [
    {
        "product_id": "PROD-F6",
        "name": "USB Microphone",
        "description": "Studio-quality USB microphone for podcasting..."
    },
    # Add 5 more products
]

# Then test semantic queries:
# "I want to start podcasting" → should find microphone
# "recording equipment" → should find microphone + webcam
```

### Exercise 6: Persistent Vector Storage

Modify vector_store.py to save embeddings to disk:

```python
# Change from in-memory to file-based
vectorstore = QdrantVectorStore.from_documents(
    documents=all_documents,
    embedding=embeddings,
    path="./qdrant_storage",  # Persistent storage!
    collection_name="techstore_products"
)

# Now vectors persist between runs
```

### Exercise 7: Hybrid Search

Combine keyword and semantic search:

```python
@tool
def search_products_hybrid(query: str, min_price: float = 0, max_price: float = 1000) -> str:
    """Search products with both semantic matching AND price filtering."""
    # 1. Semantic search
    results = search_similar_products(VECTORSTORE, query, k=10)
    
    # 2. Filter by price range
    filtered = [r for r in results if min_price <= r['price'] <= max_price]
    
    return json.dumps(filtered[:3])
```

---

## 🔧 Troubleshooting

### Issue: "Model not found"

**Solution:**
```bash
# Pull the model explicitly
ollama pull llama3.1:8b-instruct-q4_K_M
```

### Issue: "Connection refused" to Ollama

**Solution:**
```bash
# Start Ollama service
ollama serve

# Or check if it's running
ps aux | grep ollama
```

### Issue: Vector search returns poor results

### Issue: Agent doesn't call tools

**Possible causes:**
1. **Embedding model not downloaded** - Run `ollama pull mxbai-embed-large:latest`
2. **Not enough context in documents** - Add more detailed product descriptions
3. **Query too vague** - Try more specific queries

**Solution:**
```bash
# Verify embedding model
ollama list | grep mxbai

# Test embeddings directly
uv run python vector_store.py
```

**Possible causes:**
1. **Poor tool documentation** - Improve docstrings
2. **Temperature too high** - Set to 0 for more deterministic behavior
3. **Model limitations** - Llama 3.1 supports tool calling; earlier versions don't

### Issue: "Tool parsing error"

**Solution:**
- Ensure tool returns valid JSON strings
- Add try-catch blocks in tool implementations
- Set `handle_parsing_errors=True` in AgentExecutor

### Issue: Slow responses

**Solutions:**
1. Reduce `max_iterations` in AgentExecutor
2. Use lighter Ollama model (llama3.1:8b vs larger models)
3. Limit `num_predict` tokens
4. Optimize tool execution time

---

## 📊 Sample Test Data

### Orders Available:
| Order ID | Customer Email | Status | Total |
|----------|---------------|---------|-------|
| ORD-1001 | john.doe@email.com | delivered | $115.97 |
| ORD-1002 | jane.smith@email.com | shipped | $45.00 |
| ORD-1003 | john.doe@email.com | processing | $129.99 |
| ORD-1004 | bob.wilson@email.com | delivered | $92.98 |

### Products in Inventory:
| Product ID | Name | Stock | Price |
|-----------|------|-------|-------|
| PROD-A1 | Wireless Headphones | 45 | $89.99 |
| PROD-B2 | USB-C Cable | 120 | $12.99 |
| PROD-C3 | Laptop Stand | 8 | $45.00 |
| PROD-D4 | Mechanical Keyboard | 0 | $129.99 |
| PROD-E5 | Webcam HD | 23 | $79.99 |

---

## 🚀 Next Steps

### Expand the Agent:

1. **Add more tools:**
   - Cancel order
   - Modify shipping address
   - Add items to cart
   - Apply discount codes

2. **Enhance data layer:**
   - Connect to real database (PostgreSQL, MongoDB)
   - Add user authentication
   - Implement actual email sending

3. **Improve agent capabilities:**
   - Add RAG (Retrieval Augmented Generation) for product FAQs
   - Implement sentiment analysis for customer satisfaction
   - Add multilingual support

4. **Deploy to production:**
   - Containerize with Docker
   - Deploy to cloud (AWS, Azure, GCP)
   - Add monitoring and logging
   - Implement rate limiting

### Resources for Further Learning:

- **LangChain Docs**: https://python.langchain.com/docs/modules/agents/
- **Ollama Docs**: https://ollama.ai/docs
- **Streamlit Docs**: https://docs.streamlit.io
- **LangChain Tool Calling**: https://python.langchain.com/docs/modules/agents/tools/

---

## 📝 Key Takeaways

1. **Vector embeddings** enable semantic search - understanding meaning, not just keywords
2. **Qdrant** provides efficient similarity search for AI applications
3. **Tool calling** lets LLMs interact with external systems dynamically
4. **Good documentation** (docstrings) is critical for agent performance  
5. **Structured data** (JSON) enables reliable agent-tool communication
6. **Conversation memory** makes agents context-aware across turns
7. **Error handling** ensures graceful degradation when tools fail
8. **Local LLMs** (via Ollama) enable private, offline AI applications
9. **Hybrid approaches** (vector + traditional search) often work best

---

## 🤝 Contributing

This is a learning project! Feel free to:
- Add new tools and share them
- Improve error handling
- Enhance the UI
- Add tests
- Create new tutorials

---

## 📄 License

MIT License - Use freely for learning and commercial projects.

---

**Happy Learning! 🎉**

If you have questions or get stuck, review the heavily commented code in `customer_service_agent.py` and `streamlit_app.py`. Every function and section is explained in detail.