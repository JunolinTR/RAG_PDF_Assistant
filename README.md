
# 📄 **DocuLLaMA – RAG-based Chat with PDF (LLaMA 3 + Ollama)**

**DocuLLaMA** is a Retrieval-Augmented Generation (RAG)-based PDF chatbot powered by **LLaMA 3** (via Ollama). It allows you to upload PDF files, extracts and stores the content in a **FAISS vector database**, and then lets you ask questions. The chatbot retrieves relevant text chunks and answers based **only on the provided PDFs**.

---

## ✅ **Features**

* 📥 **Upload multiple PDFs** at once.
* 🔍 **Retrieval-Augmented Generation (RAG)** for context-aware answers.
* 🧠 **FAISS-based vector search** for fast and accurate chunk retrieval.
* 🤖 **LLaMA 3 (via Ollama)** for generating human-like answers.
* 🖥️ **Streamlit Web UI** for easy interaction.

---

## 🛠 **Tech Stack**

* **Frontend/UI:** Streamlit
* **PDF Processing:** PyPDF2
* **Text Chunking:** LangChain `RecursiveCharacterTextSplitter`
* **Embeddings:** SpaCy (`en_core_web_sm`)
* **Vector Database:** FAISS
* **LLM:** LLaMA 3 via Ollama

---

## 🚀 **Installation & Setup**

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/docullama.git
cd docullama
```

### **2. Install Dependencies**

Make sure you have **Python 3.9+** installed.

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, create one with:

```
streamlit
PyPDF2
langchain
langchain-community
spacy
faiss-cpu
ollama
```

### **3. Download SpaCy Model**

```bash
python -m spacy download en_core_web_sm
```

### **4. Install & Run Ollama**

* [Download Ollama](https://ollama.ai/download) and install it.
* Pull LLaMA 3 model:

```bash
ollama pull llama3
```

### **5. Run the App**

```bash
streamlit run app.py
```

Open the app in your browser (default: **[http://localhost:8501](http://localhost:8501)**).

---

## 💻 **How to Use**

1. **Upload PDF(s):** Use the sidebar to upload one or more PDF files.
2. **Process PDFs:** Click **"Submit & Process"** to extract and store chunks in FAISS.
3. **Ask Questions:** Type your question in the text input field.
4. **Get Answers:** LLaMA 3 will answer based on the retrieved context from your PDFs.

---

## 🖼 **Project Workflow**

```
PDF Upload → Extract Text → Chunking → Embedding (SpaCy) → FAISS Vector Store
→ Retrieve Relevant Chunks → Pass to LLaMA 3 → Generate Answer
```
<img width="1919" height="867" alt="image" src="https://github.com/user-attachments/assets/b42d49be-a19c-4a56-ac0b-98cc452ebb84" />
