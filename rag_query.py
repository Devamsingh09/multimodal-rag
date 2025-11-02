
import argparse
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import Qdrant
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient

import config

# ---  Conversational Memory ---
# Creating a directory to store session histories
SESSION_DIR = Path("./chat_sessions")
SESSION_DIR.mkdir(exist_ok=True)

def load_memory(session_id: str) -> list:
    """Loads chat history from a session file."""
    session_file = SESSION_DIR / f"{session_id}.json"
    if not session_file.exists():
        return []
    
    with open(session_file, 'r') as f:
        history_dicts = json.load(f)
        
    messages = []
    for msg in history_dicts:
        if msg['type'] == 'human':
            messages.append(HumanMessage(content=msg['content']))
        elif msg['type'] == 'ai':
            messages.append(AIMessage(content=msg['content']))
    return messages

def save_memory(session_id: str, history_messages: list):
    """Saves chat history to a session file."""
    session_file = SESSION_DIR / f"{session_id}.json"
    history_dicts = []
    for msg in history_messages:
        history_dicts.append({
            "type": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content
        })
    with open(session_file, 'w') as f:
        json.dump(history_dicts, f, indent=2)

def format_docs(docs: list) -> str:
    """Formats retrieved documents into a string."""
    return "\n\n".join(
        f"--- Source (Page {d.metadata.get('page_number', 'N/A')}, "
        f"Type: {d.metadata.get('type', 'text')}) ---\n"
        f"{d.page_content}"
        for d in docs
    )

def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument("--question", type=str, required=True, help="The question to ask.")
    parser.add_argument("--summarize", action="store_true", help="Enable summarization of retrieved context.")
    parser.add_argument("--session_id", type=str, default=None, help="Session ID to maintain conversational memory.")
    
    args = parser.parse_args()

    # 1. Initializing models, vector store, and retriever
    try:
        llm = ChatOllama(
            model=config.LLM_MODEL, 
            base_url=config.OLLAMA_BASE_URL
        )
        embeddings = OllamaEmbeddings(
            model=config.TEXT_EMBEDDING_MODEL, 
            base_url=config.OLLAMA_BASE_URL
        )
        
        # Connecting to the existing local Qdrant
        qdrant_client = QdrantClient(url=config.QDRANT_URL)
        
        vector_store = Qdrant(
            client=qdrant_client,
            collection_name=config.COLLECTION_NAME,
            embeddings=embeddings
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 chunks
        
        print("Ollama and Qdrant connections established.")
    except Exception as e:
        print(f"Error initializing models or Qdrant: {e}")
        print("Please ensure 'python setup_pipeline.py' was run successfully "
              "and that Ollama is running.")
        return

    # 2. Loading conversational memory (if session_id is provided)
    chat_history = []
    if args.session_id:
        print(f"--- Loading chat history for session: {args.session_id} ---")
        chat_history = load_memory(args.session_id)

    # 3. Define Chains

    # --- Summarization Chain (Requirement ) ---
    summarizer_prompt = ChatPromptTemplate.from_template(
        "Concisely summarize the following context, which was retrieved to answer a user's question. "
        "Focus on the key facts, formulas, and definitions.\n\n"
        "CONTEXT:\n{context}"
    )
    summarizer_chain = summarizer_prompt | llm | StrOutputParser()

    # --- RAG Chain (Requirement 2.2) ---
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert assistant for a 10th-grade mathematics textbook. "
         "Answer the user's question based *only* on the following context. "
         "If the context does not contain the answer, state that. "
         "When possible, cite the page number from the source metadata."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "CONTEXT:\n{context}\n\nQUESTION:\n{question}")
    ])

    # 4. Defining the Full Chain (with LCEL)

    # Retrieves documents based on the question
    retrieval_chain = (
        RunnableLambda(lambda x: x["question"]) | 
        retriever | 
        RunnableLambda(format_docs)
    )

    # This is the main chain that orchestrates everything
    full_rag_chain = (
        RunnablePassthrough.assign(
            # Load history
            chat_history=RunnableLambda(lambda x: chat_history),
            # Retrieve context
            context=retrieval_chain
        ) |
        RunnablePassthrough.assign(
            # Conditionally summarizing the retrieved context
            summary=RunnableLambda(
                lambda x: summarizer_chain.invoke({"context": x["context"]})
                if args.summarize else "N/A"
            )
        ) |
        RunnablePassthrough.assign(
            # Passing the full context, history, and question to the final prompt
            answer=(rag_prompt | llm | StrOutputParser())
        )
    )

    # 5. Invoking the chain and print results
    
    print(f"--- Querying for: '{args.question}' ---")
    result = full_rag_chain.invoke({"question": args.question})

    # --- Summarization Demonstration ---
    if args.summarize:
        print("\n### 1. Retrieved Context Summary ###")
        print(result["summary"])
        print("-" * 30)
        print("\n### 2. Final RAG Answer ###")
    else:
        print("\n### Final RAG Answer ###")

    print(result["answer"])
    
    # --- Sources Demonstration ---
    print("\n" + "-" * 30)
    print("### Retrieved Sources ###")
    print(result["context"])
    print("-" * 30)

    # 6. Saving memory for next turn (if session_id is provided)
    if args.session_id:
        chat_history.append(HumanMessage(content=args.question))
        chat_history.append(AIMessage(content=result["answer"]))
        save_memory(args.session_id, chat_history)
        print(f"\n(Context saved to session: {args.session_id})")

if __name__ == "__main__":
    main()
