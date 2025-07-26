# multilingual_RAG_10minschool
A Multilingual Retrieval-Augmented Generation (RAG) system designed for Bangla and English question answering using OCR-extracted text from PDFs. The pipeline uses Tesseract OCR for Bangla text extraction, multilingual embeddings , FAISS for vector storage, Qwen2.5-3B and FastAPI implementation as the language model. 


### ✅ **Setup Guide for multilingual\_RAG\_10minschool**

#### **1. Clone the Repository (if from GitHub)**

```bash
git clone https://github.com/<your-username>/multilingual_RAG_10minschool.git
cd multilingual_RAG_10minschool
```

---

#### **2. Install Dependencies**

For **Colab or kaggle notebook**, simply run:

```bash
!pip install -U bitsandbytes transformers accelerate sentence-transformers faiss-cpu fastapi uvicorn nest_asyncio pyngrok
!apt-get install -y poppler-utils tesseract-ocr tesseract-ocr-ben
!pip install pdf2image pytesseract
```

If running **locally**, create a `requirements.txt` file with:

```
bitsandbytes
transformers
accelerate
sentence-transformers
faiss-cpu
pdf2image
pytesseract
fastapi
uvicorn
nest_asyncio
pyngrok
```

Then install:

```bash
pip install -r requirements.txt
```

---

#### **3. Run the Notebook**

Open `multilingual_RAG_10minschool.ipynb` in **Google Colab** or Jupyter and execute cells in order just as in the notebook


---

#### **4. Start the API**

In the notebook, run the FastAPI cell:

```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

Get your public URL using ngrok:

```python
from pyngrok import ngrok
public_url = ngrok.connect(8000)
print(f"API Docs available at: {public_url}/docs")
```

---

#### **5. Test the API**

Open the printed `/docs` link → Click **POST /ask** → Try queries like:

```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

Or:

```json
{
  "query": "Who is referred to as the good man according to Anupomer ?"
}
```

---


---

### ✅ **Used Tools, Libraries, and Packages**

#### **1. OCR & PDF Handling**

* **`pdf2image`** – Convert PDF pages to images for OCR.
* **`pytesseract`** – Tesseract OCR engine for extracting Bangla text from images.
* **`poppler-utils`** – Required for `pdf2image` to work.

#### **2. Text Preprocessing & Chunking**

* **`re`** (Python built-in) – For text cleaning and normalization.

#### **3. Embeddings & Semantic Search**

* **`sentence-transformers`** – Used `intfloat/multilingual-e5-base` for multilingual embeddings (Bangla + English).
* **`faiss-cpu`** – Facebook AI Similarity Search, for storing and retrieving embeddings efficiently.

#### **4. Language Model (LLM)**

* **`transformers`** – Hugging Face library to load and run `Qwen/Qwen2.5-3B-Instruct` model.
* **`bitsandbytes`** – Enables 4-bit quantization for large models (memory efficient).
* **`accelerate`** – Optimized loading of models with device mapping.

#### **5. REST API**

* **`fastapi`** – For building REST API endpoints (`/ask`).
* **`uvicorn`** – ASGI server to run the FastAPI app.
* **`nest_asyncio`** – Allows FastAPI to run inside Jupyter/Colab without conflict.
* **`pyngrok`** – Creates a public URL for the API in Colab (via tunneling).

#### **6. Evaluation**

* **`scikit-learn`** – For `cosine_similarity` in RAG evaluation.

---



