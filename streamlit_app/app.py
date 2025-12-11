"""
Streamlit Frontend for Customer Service Agent
==============================================
Interactive web interface for the LangChain customer service agent.
Provides a chat interface with message history and agent reasoning display.
"""

import streamlit as st
from src.agents.customer_service_agent import (
    create_customer_service_agent,
    ORDERS,
    INVENTORY,
    REFUNDS
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="TechStore Customer Service",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

# Initialize session state variables
# These persist across Streamlit reruns (button clicks, input changes, etc.)

if "agent" not in st.session_state:
    # Create the agent only once per session
    st.session_state.agent = create_customer_service_agent()

if "messages" not in st.session_state:
    # Store chat history: list of {"role": "user"/"assistant", "content": "..."}
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    # Store LangChain message format for agent context
    st.session_state.chat_history = []


# =============================================================================
# SIDEBAR: Reference Information and Quick Actions
# =============================================================================

with st.sidebar:
    st.title("🛍️ TechStore Support")
    st.markdown("---")
    
    # Reference data for testing
    st.subheader("📋 Test Data Reference")
    
    with st.expander("🔢 Sample Order IDs", expanded=False):
        st.markdown("**Available orders for testing:**")
        for order in ORDERS:
            st.code(f"{order['order_id']} - {order['customer_email']}")
            st.caption(f"Status: {order['status']} | Total: ${order['total']}")
            st.markdown("---")
    
    with st.expander("📦 Sample Products", expanded=False):
        st.markdown("**Products in inventory:**")
        for product in INVENTORY:
            stock_emoji = "✅" if product['stock'] > 0 else "❌"
            st.markdown(f"{stock_emoji} **{product['name']}**")
            st.caption(f"Stock: {product['stock']} | Price: ${product['price']}")
    
    with st.expander("💳 Sample Queries", expanded=False):
        st.markdown("""
        **Try these example queries:**
        
        **🔍 Semantic Search (AI-powered):**
        1. "I need something for gaming"
        2. "Show me wireless audio equipment"
        3. "What do you have for ergonomic office setup?"
        
        **📦 Order Management:**
        4. "Can you look up order ORD-1001 for john.doe@email.com?"
        5. "What's the status of order ORD-1002?"
        
        **📋 Exact Product Lookup:**
        6. "Is the Mechanical Keyboard available?"
        7. "Do you have any USB-C cables?"
        
        **❓ Knowledge Base:**
        8. "What's your return policy?"
        9. "How long does shipping take?"
        
        **💰 Refunds:**
        10. "I'd like to return order ORD-1001. It's defective."
        """)
    
    st.markdown("---")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()
    
    # Show refund tickets if any exist
    if REFUNDS:
        st.markdown("---")
        st.subheader("🎫 Active Refund Tickets")
        for refund in REFUNDS:
            with st.container():
                st.caption(f"**{refund['ticket_id']}**")
                st.caption(f"Order: {refund['order_id']}")
                st.caption(f"Status: {refund['status']}")
                st.markdown("---")


# =============================================================================
# MAIN CHAT INTERFACE
# =============================================================================

st.title("💬 Customer Service Chat")
st.caption("Powered by LangChain + Ollama (Llama 3.1)")

# Display chat history
# Iterate through stored messages and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
# This creates a text input at the bottom of the page
if prompt := st.chat_input("Type your question here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        # Show a spinner while the agent is thinking/calling tools
        with st.spinner("Thinking..."):
            try:
                # Invoke the agent with the current input and chat history
                response = st.session_state.agent.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                
                # Extract the final output
                assistant_response = response["output"]
                
                # Display the response
                st.markdown(assistant_response)
                
                # Add to chat history for display
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
                # Update LangChain chat history for context
                # This helps the agent remember previous conversation
                from langchain_core.messages import HumanMessage, AIMessage
                st.session_state.chat_history.extend([
                    HumanMessage(content=prompt),
                    AIMessage(content=assistant_response)
                ])
                
            except Exception as e:
                # Handle any errors gracefully
                error_message = f"❌ Error: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.caption("""
ℹ️ **About this demo:**  
This is a LangChain-powered customer service agent using Ollama's Llama 3.1 model.  
The agent can look up orders, check inventory, and process refunds using tool calling.  
All data is simulated for demonstration purposes.
""")