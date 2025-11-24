# chatbot.py
from dotenv import load_dotenv
load_dotenv()

import os
import logging
import json
import uuid
import time
import requests
import PyPDF2
import smtplib
import bcrypt
from pdf2image import convert_from_bytes
import pytesseract
from serpapi import GoogleSearch
from bs4 import BeautifulSoup
from pinecone import Pinecone, ServerlessSpec
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from pydantic import BaseModel, EmailStr, field_validator
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Optional, Dict
from slowapi import Limiter
from slowapi.util import get_remote_address
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def log_error(error: Exception, context: Optional[Dict] = None):
    log_data = {
        "error": str(error),
        "context": context or {},
    }
    logging.error(json.dumps(log_data))

# Pydantic settings model
class Settings(BaseModel):
    OPENROUTER_API_KEY: str
    PINECONE_API_KEY: str
    SERPAPI_KEY: str
    EMAIL_USER: str
    EMAIL_PASS: str
    SECURE_PASSWORD_HASH: str
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587

    class Config:
        env_file = ".env"       

settings = Settings(**{k: v for k, v in os.environ.items() if k in Settings.model_fields})

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI and rate limiter
app = FastAPI(
    title="MyPersonalAI",
    description="Advanced AI Assistant with Integrated Services",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (adjust for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Pinecone initialization
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
index_name = "mypersonalai"
model_name = "intfloat/e5-large-v2"

# Create Pinecone index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    time.sleep(10)  # Wait for index initialization

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name=model_name)
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings,
    text_key="text"
)

# Security setup
security = HTTPBasic()

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(input_password: str, stored_hash: str) -> bool:
    return bcrypt.checkpw(input_password.encode(), stored_hash.encode())

# Pydantic models
class ChatRequest(BaseModel):
    user_input: str

    @field_validator('user_input')
    def validate_user_input(cls, value):
        if len(value.strip()) < 1:
            raise ValueError('user_input cannot be empty')
        if len(value) > 1000:
            raise ValueError('user_input too long')
        return value
    
class SearchRequest(BaseModel):
    query: str    

class EmailRequest(BaseModel):
    to: EmailStr
    subject: str
    message: str

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

@app.post("/chat", summary="Chat with AI", response_description="AI response")
@limiter.limit("5/minute")
async def chat_with_ai(
    request: Request,
    chat_request: ChatRequest,
    credentials: HTTPBasicCredentials = Depends(security)
):
    try:
        logging.info("Authentication check started")
        if not verify_password(credentials.password, settings.SECURE_PASSWORD_HASH):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        logging.info("Authentication successful")
        user_message = chat_request.user_input

        logging.info("Pinecone similarity search started")
        similar_results = vectorstore.similarity_search(user_message, k=3)
        context = "\n".join([doc.page_content for doc in similar_results])
        logging.info(f"Context: {context}")

        logging.info("Preparing OpenRouter request")
        payload = {
            "model": "deepseek/deepseek-r1:free",
            "messages": [
                {"role": "system", "content": "You are an advanced AI assistant."},
                {"role": "user", "content": f"Context: {context}\n\n{user_message}"}
            ],
            "temperature": 0.75,
            "max_tokens": 500
        }

        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        logging.info("Sending request to OpenRouter")
        response = requests.post(OPENROUTER_API_URL, json=payload, headers=headers)
        response.raise_for_status()

        logging.info("OpenRouter response received")
        assistant_response = response.json()["choices"][0]["message"]["content"]

        logging.info("Storing conversation in Pinecone")
        combined_text = f"User: {user_message}\nAI: {assistant_response}"
        vectorstore.add_texts(
            texts=[combined_text],
            ids=[str(uuid.uuid4())],
            metadatas=[{'text': combined_text}]
        )

        return {"response": assistant_response}

    except Exception as e:
        logging.error(f"Full error details: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search-web", summary="Web Search", response_description="Search results")
@limiter.limit("10/minute")
async def search_web(request: Request,search_request: SearchRequest):  # Use Pydantic model
    """Perform a web search using Google via SerpAPI"""
    try:
        params = {
            "q": search_request.query,  # Access via model
            "api_key": settings.SERPAPI_KEY,
            "engine": "google",
            "num": 5
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        search_results = []
        if "organic_results" in results:
            for result in results["organic_results"]:
                search_results.append({
                    "title": result.get("title", "No Title"),
                    "link": result.get("link", "No Link"),
                    "snippet": result.get("snippet", "No Snippet")
                })

        return {"results": search_results}

    except Exception as e:
        log_error(e, {"query": search_request.query})
        raise HTTPException(status_code=500, detail="Search failed")
    

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"   

@app.post("/read-pdf", summary="PDF Text Extraction", response_description="Extracted text")
@limiter.limit("3/minute")
async def read_pdf(request: Request, file: UploadFile = File(...)):
    """Extract text from PDF files with OCR fallback"""
    try:
        logging.info(f"Processing file: {file.filename}")
        
        # Read contents
        contents = await file.read()
        logging.info(f"File size: {len(contents)} bytes")

        text = ""
        try:
            # Attempt PyPDF2 extraction
            logging.info("Attempting PyPDF2 extraction")
            reader = PyPDF2.PdfReader(contents)
            text = "".join([page.extract_text() for page in reader.pages])
            
            if not text.strip():
                logging.warning("PyPDF2 returned empty text")
                raise ValueError("No text found in PDF")
                
            logging.info("PyPDF2 extraction successful")
            
        except Exception as pdf_error:
            logging.warning(f"PyPDF2 failed: {str(pdf_error)}")
            
            # Fallback to OCR
            logging.info("Attempting OCR fallback")
            try:
                # Convert PDF to images with Poppler
                images = convert_from_bytes(
                    contents,
                    poppler_path = r'C:\poppler\poppler-24.08.0\Library\bin'# Windows example
                    # poppler_path='/usr/bin'  # Linux/Mac
                )
                logging.info(f"Converted {len(images)} pages to images")
                
                # Extract text from images
                text = "\n".join([pytesseract.image_to_string(img) for img in images])
                logging.info("OCR completed successfully")
                
            except Exception as ocr_error:
                logging.error(f"OCR failed: {str(ocr_error)}", exc_info=True)
                raise RuntimeError(f"OCR processing failed: {str(ocr_error)}") from ocr_error

        return {"content": text[:5000]}

    except Exception as e:
        logging.error(f"Final error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))  # Return specific error message
    
@app.post("/send-email", summary="Send Email", response_description="Email status")
@limiter.limit("2/minute")
async def send_email(request: Request, email_request: EmailRequest):  # <-- Add request parameter
    """Send emails via SMTP with improved security"""
    try:
        server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT)
        server.starttls()
        server.login(settings.EMAIL_USER, settings.EMAIL_PASS)
        
        email_message = f"Subject: {email_request.subject}\n\n{email_request.message}"
        server.sendmail(settings.EMAIL_USER, email_request.to, email_message)
        server.quit()
        
        return {"status": "Email sent successfully"}

    except Exception as e:
        log_error(e, {"recipient": request.to})
        raise HTTPException(status_code=500, detail="Email sending failed")

if __name__== "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080)