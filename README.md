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

---

### ✅ **Sample Queries and Outputs**

#### **Bangla Query Example**

**Input:**

```
অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
```

**Output:**

```
শস্তুনাথ
```

---

#### **Another Bangla Query**

**Input:**

```
কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
```

**Output:**

```
মামাকে
```

---

#### **English Query Example**

**Input:**

```
Was his Anupam's marriage completed? 
```

**Output:**

```
Based on the information provided, it seems that Anupam's marriage was not completed.
```

---



### ✅ **Evaluation Matrix**

| **Query**                                       | **Expected Answer** | **Model Answer**              | **Grounded** | **Relevance (Cosine)** |
| ----------------------------------------------- | ------------------- | ----------------------------- | ------------ | ---------------------- |
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?           | শস্তুনাথ            | (গ) সুপুরুষ বটে- কে? শস্তুনাথ | ✅            | 0.81                   |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে              | মামাকে                        | ✅            | 0.80                   |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?        | ১৫ বছর              | ১৬/১৭ বছর                     | ✅            | 0.82                   |

---

#### **Evaluation Summary**

```
Grounded Answers: 3/3
Relevant Retrievals: 3/3
```

* **Groundedness:** All answers were present in the retrieved context.
* **Relevance:** Average cosine similarity > 0.80, indicating highly relevant chunks.
* **Accuracy Observation:** For the third query, the model answered approximately (16/17 years vs. 15 years), showing the model’s reasoning but not exact matching.

---



---

## ✅ **API Documentation**

### **Overview**

This API provides a REST interface for querying the RAG system

---

### **Base URL**

```
https://<your-ngrok-id>.ngrok-free.app
```

---

### **Endpoint**

```
POST /ask
```

---

### **Request Details**

* **Method:** `POST`
* **Headers:**

  ```
  Content-Type: application/json
  ```
* **Request Body Example:**

```json
{
  "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
}
```

---

### **Response Details**

* **Format:** `application/json`
* **Example Response:**

```json
{
  "question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
  "answer": "মামাকে"
}
```

---

### **How to Test the API**

#### ✅ **1. Swagger UI**

* Visit:

```
https://<your-ngrok-id>.ngrok-free.app/docs
```

* Click **POST /ask → Try it out → Enter query → Execute**
* View the response in JSON format.

---

#### ✅ **2. Curl Command**

```bash
curl -X 'POST' \
  'https://<your-ngrok-id>.ngrok-free.app/ask' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "Who is referred to as the good man in Anupomer language?"
}'
```

---

### **Authentication**

Ngrok requires authentication for generating a public link:

1. Sign up at [ngrok dashboard](https://dashboard.ngrok.com/signup).
2. Copy your **Authtoken** from:

```
https://dashboard.ngrok.com/get-started/your-authtoken
```

3. Add in Colab:

```bash
!ngrok config add-authtoken YOUR_AUTHTOKEN
```

---

### **Screenshot API**
<img width="668" height="386" alt="api_UI_screenshot" src="https://github.com/user-attachments/assets/4a2bff16-ed32-4ed0-913a-1cf8a64d73a1" />






