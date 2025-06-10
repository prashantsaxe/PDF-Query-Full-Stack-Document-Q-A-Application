from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
from datetime import datetime
import os
import uuid
import fitz  # PyMuPDF
from typing import List, Optional
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.schema import Document
import json
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = "sqlite:///./pdf_qa_app.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class DocumentModel(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    original_filename = Column(String)
    file_path = Column(String)
    upload_date = Column(DateTime, default=datetime.utcnow)
    file_size = Column(Integer)
    page_count = Column(Integer)
    text_content = Column(Text)
    embedding_path = Column(String, nullable=True)

class QuestionModel(Base):
    __tablename__ = "questions"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, index=True)
    question = Column(Text)
    answer = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    processing_time = Column(Integer)  # in milliseconds

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models for API
class DocumentResponse(BaseModel):
    id: int
    filename: str
    original_filename: str
    upload_date: datetime
    file_size: int
    page_count: int
    
    class Config:
        from_attributes = True

class QuestionRequest(BaseModel):
    document_id: int
    question: str

class QuestionResponse(BaseModel):
    id: int
    document_id: int
    question: str
    answer: str
    timestamp: datetime
    processing_time: int
    
    class Config:
        from_attributes = True

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int

# FastAPI app initialization
app = FastAPI(title="PDF Q&A API", description="API for uploading PDFs and asking questions", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIRECTORY = "uploaded_pdfs"
EMBEDDINGS_DIRECTORY = "embeddings"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(EMBEDDINGS_DIRECTORY, exist_ok=True)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# NLP Processing Class
class PDFProcessor:
    def __init__(self):
        # Initialize OpenAI (you'll need to set OPENAI_API_KEY environment variable)
        self.embeddings = OpenAIEmbeddings()
        self.llm = OpenAI(temperature=0.7)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> tuple[str, int]:
        """Extract text from PDF and return text content and page count"""
        try:
            doc = fitz.open(pdf_path)
            text_content = ""
            page_count = len(doc)
            
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                text_content += page.get_text()
            
            doc.close()
            return text_content, page_count
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    
    def create_embeddings(self, text_content: str, document_id: int) -> str:
        """Create and save embeddings for the document"""
        try:
            # Split text into chunks
            texts = self.text_splitter.split_text(text_content)
            
            # Create documents
            documents = [Document(page_content=text) for text in texts]
            
            # Create FAISS vector store
            vectorstore = FAISS.from_documents(documents, self.embeddings)
            
            # Save embeddings
            embedding_path = os.path.join(EMBEDDINGS_DIRECTORY, f"doc_{document_id}")
            vectorstore.save_local(embedding_path)
            
            return embedding_path
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")
    
    def answer_question(self, question: str, embedding_path: str) -> str:
        """Answer question using the document embeddings"""
        try:
            # Load embeddings
            vectorstore = FAISS.load_local(embedding_path, self.embeddings)
            
            # Search for relevant documents
            docs = vectorstore.similarity_search(question, k=3)
            
            # Create QA chain
            chain = load_qa_chain(self.llm, chain_type="stuff")
            
            # Get answer
            response = chain.run(input_documents=docs, question=question)
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")

# Initialize PDF processor
pdf_processor = PDFProcessor()

# API Endpoints
@app.get("/")
async def root():
    return {"message": "PDF Q&A API is running"}

@app.post("/upload", response_model=DocumentResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a PDF document"""
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        filename = f"{file_id}.pdf"
        file_path = os.path.join(UPLOAD_DIRECTORY, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Extract text from PDF
        text_content, page_count = pdf_processor.extract_text_from_pdf(file_path)
        
        # Create database record
        db_document = DocumentModel(
            filename=filename,
            original_filename=file.filename,
            file_path=file_path,
            file_size=file_size,
            page_count=page_count,
            text_content=text_content
        )
        
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        
        # Create embeddings asynchronously (in background)
        try:
            embedding_path = pdf_processor.create_embeddings(text_content, db_document.id)
            db_document.embedding_path = embedding_path
            db.commit()
        except Exception as e:
            logger.error(f"Error creating embeddings for document {db_document.id}: {str(e)}")
            # Continue without embeddings - we can create them later
        
        return DocumentResponse.from_orm(db_document)
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        # Clean up file if it was created
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/documents", response_model=DocumentListResponse)
async def get_documents(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get list of uploaded documents"""
    
    documents = db.query(DocumentModel).offset(skip).limit(limit).all()
    total = db.query(DocumentModel).count()
    
    return DocumentListResponse(
        documents=[DocumentResponse.from_orm(doc) for doc in documents],
        total=total
    )

@app.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get specific document details"""
    
    document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse.from_orm(document)

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    """Ask a question about a document"""
    
    start_time = datetime.utcnow()
    
    # Get document
    document = db.query(DocumentModel).filter(DocumentModel.id == request.document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if embeddings exist
    if not document.embedding_path or not os.path.exists(document.embedding_path):
        # Create embeddings if they don't exist
        try:
            embedding_path = pdf_processor.create_embeddings(document.text_content, document.id)
            document.embedding_path = embedding_path
            db.commit()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating embeddings: {str(e)}")
    
    # Get answer
    try:
        answer = pdf_processor.answer_question(request.question, document.embedding_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    
    # Calculate processing time
    processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    # Save question and answer
    db_question = QuestionModel(
        document_id=request.document_id,
        question=request.question,
        answer=answer,
        processing_time=processing_time
    )
    
    db.add(db_question)
    db.commit()
    db.refresh(db_question)
    
    return QuestionResponse.from_orm(db_question)

@app.get("/documents/{document_id}/questions", response_model=List[QuestionResponse])
async def get_document_questions(
    document_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get questions and answers for a specific document"""
    
    # Check if document exists
    document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    questions = (
        db.query(QuestionModel)
        .filter(QuestionModel.document_id == document_id)
        .order_by(QuestionModel.timestamp.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    
    return [QuestionResponse.from_orm(q) for q in questions]

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document and its associated data"""
    
    document = db.query(DocumentModel).filter(DocumentModel.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    try:
        # Delete physical file
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
        
        # Delete embeddings
        if document.embedding_path and os.path.exists(document.embedding_path):
            shutil.rmtree(document.embedding_path)
        
        # Delete questions
        db.query(QuestionModel).filter(QuestionModel.document_id == document_id).delete()
        
        # Delete document record
        db.delete(document)
        db.commit()
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)