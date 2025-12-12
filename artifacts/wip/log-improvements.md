## Current state of logging



Logs come from two sources:

1) LangChain AgentExecutor verbose logging
When `verbose=True` is set in the AgentExecutor, LangChain logs execution steps:
- "Entering new AgentExecutor chain..."
- "Invoking: `search_products_semantic` with `{...}`"
- "Finished chain."

This is configured in `src/agents/customer_service_agent.py` at line 406:

```406:406:src/agents/customer_service_agent.py
        verbose=True,  # Print detailed execution steps
```

2) Custom print statements in the vector store
The emoji-prefixed messages come from `print()` statements in `src/vectorstore/vector_store.py`:
- Line 164: "🔧 Initializing embeddings model..."
- Line 172: "📄 Preparing documents for embedding..."
- Line 215: "📦 Embedding {len} documents..."
- Line 226: "💾 Using persistent storage at: ..."
- Line 250: "✅ Vector store initialized with {len} documents!"

These run when the vector store is initialized, which happens the first time `search_products_semantic` is called (see line 189 in `customer_service_agent.py`).

To control logging:
- Disable LangChain verbose logging: set `verbose=False` in the AgentExecutor.
- Replace print statements: use Python's `logging` module instead of `print()`.

Should I update the code to use proper logging instead of print statements, or disable the verbose logging?