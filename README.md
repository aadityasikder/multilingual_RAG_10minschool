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



---

### ✅ **Assignment Questions and Answers (Paragraph Style)**

---

### **1. What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?**

To extract text from the Bangla PDF, we used **Tesseract OCR** via the `pytesseract` library along with `pdf2image`. The reason for this approach was that Bangla text breaks into disjoint characters or has incorrect ligatures (For example: 'নািীযকামলঠিক, র্কন্তুদুব্িলন - কলযাণীিিীব্নচর্িতদ্বািাপ্রর্তজিতএইসতযঅনুধাব্নকিকত পািকব্।') when extracted using common parsers like PyMuPDF or PDFMiner. With OCR, the text was recognized directly from rendered images, preserving the language structure better. 

---

### **2. What chunking strategy did you choose (e.g., paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?**

We adopted a **character-based chunking strategy**, where each chunk was set to 1,700 characters with an overlap of 300 characters. This approach was necessary because OCR output lacks reliable paragraph boundaries, making paragraph-based or sentence-based chunking less accurate. The overlapping strategy ensured that if a question referred to content spanning two chunks, it would still appear in the retrieved context. This method works well for semantic retrieval because it provides a balanced chunk size not too large to lose focus and not too small to lose context.

---

### **3. What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?**

We used the **intfloat/multilingual-e5-base** model for generating text embeddings. This model was chosen because it supports both Bangla and English, which was essential for handling multilingual queries. It is specifically optimized for retrieval tasks, meaning that semantically similar texts, even in different languages, are embedded close to each other in the vector space. This allows for meaningful comparison between user queries and document chunks, regardless of language differences.

---

### **4. How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?**

We stored all embeddings in a **FAISS index (IndexFlatL2)** and used **L2 distance** for similarity search. This method was selected because Faiss is a library for efficient similarity search and clustering of dense vectors from facebook. When embeddings are normalized, L2 distance effectively behaves like cosine similarity, which is widely used for semantic retrieval. This setup ensures that queries are compared with the most relevant document chunks based on semantic similarity, not just keyword overlap.

---

### **5. How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?**

We ensured meaningful comparison by using a multilingual embedding model and introducing overlapping chunks so that relevant context is not lost at chunk boundaries. Additionally, the retrieval step uses semantic similarity instead of keyword matching, allowing queries and chunks with similar meaning to align, even if their wording differs. If the query is vague or lacks sufficient context, the system may retrieve less relevant chunks or fail to find the correct answer. In such cases, our model is instructed to respond with a fallback message—“প্রদত্ত তথ্য থেকে উত্তর পাওয়া যায়নি” in Bangla or “Answer not found in context” in English. 

---

### **6. Do the results seem relevant? If not, what might improve them (e.g., better chunking, better embedding model, larger document)?**

The results were generally relevant and close to correct, but not always perfectly accurate or structured as a human might expect, as shown in our evaluation where all answers were grounded in the retrieved context and had an average cosine similarity of around 0.81. However, there were slight discrepancies in numeric details for example, the model answered “16/17 years” instead of “15 years,” which indicates approximate reasoning. 
To improve the results, the following steps can be taken:

* **Use higher-parameter models:** Larger models like LLaMA-3 or Gemma would offer better reasoning and accuracy, provided sufficient GPU resources are available.
* **Fine-tune models on similar datasets:** Training on Bangla-language QA datasets would allow the model to better understand context, idioms, and cultural nuances.
* **Experiment with chunking and retrieval parameters:** Adjusting chunk size, overlap, and number of retrieved chunks can help preserve more context and improve semantic retrieval accuracy.
* **Introduce re-ranking or hybrid retrieval:** Combining semantic search with keyword-based retrieval and applying re-ranking could further increase relevance.

---




