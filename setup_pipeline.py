import os
import base64
import warnings
from pathlib import Path
import io

from PIL import Image
from langchain_community.vectorstores import Qdrant
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from unstructured.partition.pdf import partition_pdf

import config

# Suppressing common warnings from unstructured
warnings.filterwarnings("ignore", category=UserWarning)

def encode_image_to_base64(image: Image.Image) -> str:
    """Encodes a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def summarize_image(image_bytes: bytes, llm_vision: ChatOllama) -> str:
    """
    Uses a local VLM (LlaVA) to generate a summary of an image.
    This summary is what gets embedded.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        # Resizing for faster processing, LLaVA can handle smaller images
        pil_image.thumbnail((1024, 1024))
        
        img_b64 = encode_image_to_base64(pil_image)
        
        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail. What mathematical concepts, "
                        "diagrams, or formulas are visible? Be descriptive. This "
                        "description will be used for a search index."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{img_b64}",
                },
            ]
        )
        
        print("  > Summarizing image with LlaVA...")
        response = llm_vision.invoke([msg])
        print(f"  > Summary: {response.content[:70]}...")
        return response.content
    except Exception as e:
        print(f"Error summarizing image: {e}")
        return "Failed to summarize image"

def main():
    print(f"Starting pipeline setup for {config.PDF_PATH}...")
    
    if not Path(config.PDF_PATH).exists():
        print(f"Error: PDF file not found at {config.PDF_PATH}")
        return

    # 1. Initializing models from Ollama
    try:
        embeddings = OllamaEmbeddings(
            model=config.TEXT_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        llm_vision = ChatOllama(
            model=config.IMAGE_SUMMARY_MODEL,
            base_url=config.OLLAMA_BASE_URL
        )
        # Testing connection
        llm_vision.invoke("test")
        print("Ollama models initialized successfully.")
    except Exception as e:
        print(f"Error initializing Ollama models: {e}")
        print("Please ensure Ollama is running and you have pulled "
              f"'{config.TEXT_EMBEDDING_MODEL}' and '{config.IMAGE_SUMMARY_MODEL}'.")
        print("$ ollama pull nomic-embed-text")
        print("$ ollama pull llava")
        return

    # 2. Partitioning the PDF using unstructured
    print("Partitioning PDF with 'unstructured' (hi_res strategy)...")
    print("This may take several minutes and download models on first run...")
    
    # This is the core of the multimodal extraction.
    # 'hi_res' strategy finds text, tables, and images with better accuracy.
    try:
        elements = partition_pdf(
            filename=config.PDF_PATH,
            strategy="hi_res",  # Use hi_res strategy for better extraction
            infer_table_structure=True,
            extract_images_in_pdf=True,
        )
    except Exception as e:
        print(f"Error during PDF partitioning: {e}")
        print("This often happens if system dependencies like 'poppler' or 'tesseract' are missing.")
        return

    print("PDF partitioned. Processing elements...")
    
    documents = []
    
    # 3. Processing elements and creating Documents
    for el in elements:
        element_type = str(type(el)).split('.')[-1].replace("'", "").replace(">", "")
        
        if element_type == "Image":
            if hasattr(el, 'image_bytes') and el.image_bytes:
                # Generating a text summary for the image
                summary = summarize_image(el.image_bytes, llm_vision)
                documents.append(Document(
                    page_content=summary,
                    metadata={
                        "source": config.PDF_PATH,
                        "page_number": el.metadata.page_number,
                        "type": "image_summary"
                    }
                ))
        elif "Table" in element_type:
            # Tables are often better represented by their HTML or text
            if el.metadata.text_as_html:
                content = f"Table on page {el.metadata.page_number}:\n{el.metadata.text_as_html}"
            else:
                content = el.text
            documents.append(Document(
                page_content=content,
                metadata={
                    "source": config.PDF_PATH,
                    "page_number": el.metadata.page_number,
                    "type": "table",
                }
            ))
        elif hasattr(el, 'text') and el.text.strip():
            # This captures all text elements (Title, NarrativeText, etc.)
            documents.append(Document(
                page_content=el.text,
                metadata={
                    "source": config.PDF_PATH,
                    "page_number": el.metadata.page_number,
                    "type": element_type 
                }
            ))
            
    if not documents:
        print("No documents were extracted. Exiting.")
        return

    print(f"Processed {len(documents)} text, table, and image chunks.")
    print("Creating and populating Qdrant vector store (this will take time)...")

    # 4. Creating and populate the Qdrant vector store
    # This will store files locally in the config.QDRANT_PATH directory
    Qdrant.from_documents(
        documents,
        embeddings,
        url=config.QDRANT_URL,
        collection_name=config.COLLECTION_NAME,
        force_recreate=True # Overwrite existing collection
    )

    print("---")
    print("Qdrant collection created successfully.")
    print(f"Path: {config.QDRANT_URL}")
    print(f"Collection: {config.COLLECTION_NAME}")
    print(f"{len(documents)} documents indexed.")
    print("---")
    print("Indexing run complete. You can now run rag_query.py.")

if __name__ == "__main__":
    main()
