
# Multimodal RAG Pipeline for Educational Content

This project is an intern assignment to build a fully functional, local-first Retrieval-Augmented Generation (RAG) pipeline. The system processes a complex PDF containing text, tables, and images, and allows a user to query its content.

The pipeline is built entirely with **Python**, using **LangChain** for orchestration, **Ollama** to run all AI models locally (for generation, embeddings, and vision), and **Qdrant** (running in Docker) as the vector database.

## 🚀 Key Features

* **Multimodal Processing:** The pipeline uses the `unstructured` library to parse the PDF, extracting text, tables, and images.
* **Local-First AI:** All AI models run locally via Ollama:
    * **Generation:** `llama3`
    * **Embeddings:** `nomic-embed-text`
    * **Vision:** `llava` (or a `llava-cpu` variant) is used to generate text summaries of images, making them searchable.
* **Server-Based Database:** Qdrant is run as a Docker container to provide a stable, server-based vector store, preventing the file-locking issues common with local file paths.
* **Advanced Capabilities:**
    * **Context Summarization:** A `--summarize` flag provides a concise summary of retrieved context before the final answer.
    * **Conversational Memory:** A `--session_id` flag enables conversational caching, allowing the model to remember and answer follow-up questions.

## 🛠️ Setup Instructions

Follow these steps to set up and run the project.

### 1. Prerequisite: Install Ollama
The Ollama application serves all the local AI models.

1.  **Download & Install:** Go to [https://ollama.com](https://ollama.com) and install the application for your operating system.
2.  **Run Ollama:** Start the Ollama application. It will run in the background.
3.  **Pull Models:** Open your terminal and pull the three required models:
    ```bash
    ollama pull llama3
    ollama pull nomic-embed-text
    ollama pull llava
    ```
4.  **(VRAM TROUBLESHOOTING):** If `setup_pipeline.py` fails with a "system memory" error (as we saw in testing), your GPU doesn't have enough VRAM for `llava`. Create a CPU-only version by:
    * Creating a file named `llava-cpu.Modelfile` with this content:
        ```
        FROM llava
        PARAMETER num_gpu 0
        ```
    * Running this command in your terminal:
        ```bash
        ollama create llava-cpu -f llava-cpu.Modelfile
        ```
    * Changing the `IMAGE_SUMMARY_MODEL` in `config.py` to `"llava-cpu"`.

### 2. Prerequisite: Install Docker Desktop
Qdrant runs in a Docker container to prevent database lock errors.

1.  **Download & Install:** Go to [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/) and install it.
2.  **Run Docker:** Start the Docker Desktop application.

### 3. Prerequisite: Install System Dependencies
The `unstructured` library requires `poppler` and `tesseract` to read PDFs and images.

* **On macOS:**
    ```bash
    brew install poppler tesseract
    ```
* **On Ubuntu/Linux:**
    ```bash
    sudo apt-get update
    sudo apt-get install -y poppler-utils tesseract-ocr
    ```
* **On Windows:**
    1.  Install **Tesseract** from the [installer](https://github.com/UB-Mannheim/tesseract/wiki), making sure to check "Add to PATH".
    2.  Download **Poppler** from [GitHub](https://github.com/oschwartz10612/poppler-windows/releases/latest) and unzip it to a permanent location (e.g., `C:\Program Files\poppler-25.07.0`).
    3.  Manually add the Poppler `bin` folder (e.g., `C:\Program Files\poppler-25.07.0\Library\bin`) to your Windows "Environment Variables" `Path`.
    4.  **Restart VSCode** to apply the new PATH.

### 4. Project Setup
1.  **Clone Repository:**
    ```bash
    git clone [your-repo-link]
    cd [your-repo-name]
    ```
2.  **Move Project (if needed):** Ensure the project is on a local hard drive (like `C:\`) and **not** a pendrive, which can cause disk space errors (`No space left on device`).
3.  **Create `.env` File:** Create a `.env` file and add your Hugging Face token (used by `unstructured` to download helper models):
    ```ini
    HUGGINGFACE_TOKEN="your_hf_token_here"
    ```
4.  **Add Data:** Place your PDF file (e.g., `jemh109.pdf`) into the `data/` folder.
5.  **Create Python Environment:**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```
6.  **Install Packages:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Running the Pipeline (MANDATORY DEMONSTRATION)

This section provides the *actual console output* from running the finished project.

### Step 1: Start the Qdrant Server
Open a terminal, navigate to your project folder, and run:

```bash
docker run -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```
**Leave this terminal running.** It is your database.

### Step 2: Indexing Run
Open a **new** terminal (activate your `venv` here).

```bash
$ python setup_pipeline.py
```
**Actual Output:**
```
Starting pipeline setup for data/jemh109.pdf...
Ollama models initialized successfully.
Partitioning PDF with 'unstructured' (hi_res strategy)...
This may take several minutes and download models on first run...
Warning: No languages specified, defaulting to English.
The `max_size` parameter is deprecated and will be removed in v4.26. Please specify in `size['longest_edge'] instead`.
PDF partitioned. Processing elements...
Processed 149 text, table, and image chunks.
Creating and populating Qdrant vector store (this will take time)...
---
Qdrant collection created successfully.
Path: http://localhost:6333
Collection: jmeh_multimodal
149 documents indexed.
---
Indexing run complete. You can now run rag_query.py.
```

### Step 3: RAG Query Demonstration (Non-Conversational)
This query proves the RAG pipeline is grounded and will **not** hallucinate. The sample query is about "quadratic equations" (Chapter 4), but our PDF is "trigonometry" (Chapter 9). The pipeline correctly states the information is missing.

```bash
$ python rag_query.py --question "Explain the steps involved in solving a quadratic equation as mentioned in Chapter 4."
```
**Actual Output:**
```
Ollama and Qdrant connections established.
--- Querying for: 'Explain the steps involved in solving a quadratic equation as mentioned in Chapter 4.' ---

### Final RAG Answer ###
I apologize, but there is no mention of quadratic equations or their solution in the provided context. The topics discussed include trigonometric ratios (Page 1), and ways to use trigonometry in real-life scenarios (Page 11). Additionally, a formula is presented for finding BC given AB and cotA (Page 3), but it does not involve solving a quadratic equation.

Therefore, I cannot provide the steps involved in solving a quadratic equation as mentioned in Chapter 4 since it's not present in the provided context.

------------------------------
### Retrieved Sources ###
--- Source (Page 3, Type: NarrativeText) ---
Now let us explain the process, we have just discussed, by solving some problems.

--- Source (Page 11, Type: NarrativeText) ---
In this chapter, you have studied the following points :
...
```

### Step 4: Multimodal Retrieval Proof
This test demonstrates that content *related* to the diagrams is indexed. In this specific PDF, the text descriptions (e.g., "Example 2") are a richer and more relevant vector match than the AI-generated image summaries. The pipeline correctly retrieves this text.

```bash
$ python rag_query.py --question "What does the diagram showing an electrician on a ladder against a pole show?"
```
**Actual Output:**
```
Ollama and Qdrant connections established.
--- Querying for: 'What does the diagram showing an electrician on a ladder against a pole show?' ---

### Final RAG Answer ###
According to the context, the diagram shows the electrician reaching the point B on the pole AD (Source: Page 4). The specific details shown in the diagram are that the electrician needs to repair an electric fault on the pole at a height of 5 m and reach a point 1.3m below the top of the pole, with the ladder inclined at an angle of 60° to the horizontal.

------------------------------
### Retrieved Sources ###
--- Source (Page 4, Type: NarrativeText) ---
Solution : In Fig. 9.5, the electrician is required to reach the point B on the pole AD.

--- Source (Page 4, Type: NarrativeText) ---
Example 2 : An electrician has to repair an electric fault on a pole of height 5 m. She needs to reach a point 1.3m below the top of the pole to undertake the repair work (see Fig. 9.5).
...
```

### Step 5: Caching Proof (Conversational)
This demonstrates the conversational memory using `--session_id`.

**First Question:**
```bash
$ python rag_query.py --question "In Example 1, what is the height of the tower?" --session_id "test_01"
```
**Actual Output:**
```
Ollama and Qdrant connections established.
--- Loading chat history for session: test_01 ---
--- Querying for: 'In Example 1, what is the height of the tower?' ---

### Final RAG Answer ###
According to the context, the height of the tower is found to be 20/3 m (Source, Page 7).
...
(Context saved to session: test_01)
```

**Second Question (Follow-up):**
The context from the first question is loaded, and the pipeline correctly retrieves sources about `tan 60°` for "Example 1".

```bash
$ python rag_query.py --question "What trigonometric ratio was used to find it?" --session_id "test_01"
```
**Actual Output:**
```
Ollama and Qdrant connections established.
--- Loading chat history for session: test_01 ---
--- Querying for: 'What trigonometric ratio was used to find it?' ---

### Final RAG Answer ###
Based on the provided context, there is no mention of finding the height of a tower. It seems that the context is focused on using trigonometric ratios in right triangles to solve problems, but does not specifically address finding the height of a tower.

------------------------------
### Retrieved Sources ###
--- Source (Page 3, Type: NarrativeText) ---
To solve the problem, we choose the trigonometric ratio tan 60° (or cot 60°), as the ratio involves AB and BC.
...
(Context saved to session: test_01)
```

### Step 6: Summarization Demonstration
This test proves the `--summarize` flag works. It correctly prints a summary of the retrieved (irrelevant) context *before* printing the final answer.

```bash
$ python rag_query.py --summarize --question "What is the concept of 'Arithmetic Progression'?"
```
**Actual Output:**
```
Ollama and Qdrant connections established.
--- Querying for: 'What is the concept of 'Arithmetic Progression'?' ---

### 1. Retrieved Context Summary ###
Here are the key facts, formulas, and definitions summarized from the context:

**Key Facts:**

* Trigonometry is used to determine the height or length of an object or distance between two distant objects.
* Trigonometric ratios can be applied to real-life scenarios.

**Formulas:**

None mentioned in this context.

**Definitions:**

* Angle of elevation: The angle formed by the line of sight with the horizontal when it is above the horizontal level, i.e., when looking up at an object.
------------------------------

### 2. Final RAG Answer ###
I'm afraid I couldn't find any information about "Arithmetic Progression" in the provided context. The text only discusses trigonometric ratios and their applications in real-life scenarios...

------------------------------
### Retrieved Sources ###
--- Source (Page 1, Type: NarrativeText) ---
In the previous chapter, you have studied about trigonometric ratios. In this chapter...
...
```
````
