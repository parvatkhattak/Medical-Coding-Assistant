import os
import logging
from typing import List, Dict, Any, Optional
import json
import time
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
import nltk
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# Define custom download directory
custom_dir = "nltk_data"

# Download stopwords to the specified folder
nltk.download('stopwords', download_dir=custom_dir)
nltk.download('punkt', download_dir=custom_dir)

from nltk.corpus import stopwords
nltk.data.path.append(custom_dir)

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = [word.strip() for word in text.lower().split() if word.strip() != ""]
    return [word for word in words if word.isalnum() and word not in stop_words]

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_api_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
COLLECTION_NAME = "Medical_Coder"

# Document groups matching your process_documents.py structure
DOCUMENT_GROUPS = {
    "ICD_CODES": {
        "collection":"Medical_Coder",
        "files": ["RAG1.pdf", "RAG1_1.xlsx"],
        "description": "ICD-10 Coding Guidelines and References",
        "priority": 1
    },
    "CPT_PROCEDURES": {
        "collection":"Medical_Coder",
        "files": ["RAG2.xlsx", "RAG2_1.pdf", "RAG2_2.pdf", "RAG2_3.pdf"],
        "description": "CPT Procedure Codes and Documentation",
        "priority": 2
    },
    "MEDICAL_TERMINOLOGY": {
        "collection":"Medical_Coder",
        "files": ["RAG3.csv"],
        "description": "Medical Terminology and Definitions",
        "priority": 3
    }
}


# Initialize clients
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Initialize Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# Rate limiting tracker
class RateLimitTracker:
    def __init__(self):
        self.requests = []
        self.daily_requests = 0
        self.daily_reset = datetime.now() + timedelta(days=1)
        
    def can_make_request(self):
        now = datetime.now()
        
        # Reset daily counter if needed
        if now > self.daily_reset:
            self.daily_requests = 0
            self.daily_reset = now + timedelta(days=1)
        
        # Remove old requests (older than 1 minute)
        self.requests = [req_time for req_time in self.requests if now - req_time < timedelta(minutes=1)]
        
        # Check limits (conservative estimates for free tier)
        if self.daily_requests >= 500:  # Conservative daily limit
            return False, "Daily quota exceeded"
        if len(self.requests) >= 15:  # Conservative per-minute limit
            return False, "Per-minute quota exceeded"
        
        return True, "OK"
    
    def record_request(self):
        self.requests.append(datetime.now())
        self.daily_requests += 1

rate_limiter = RateLimitTracker()

class GeminiEmbeddings:
    """Wrapper for Gemini embeddings with enhanced error handling"""
    
    def __init__(self, model_name: str = 'models/text-embedding-004'):
        self.model_name = model_name
        self.rate_limit_delay = 1.0  # Increased delay
        
    def embed_query(self, text: str) -> List[float]:
        """Embed a query using Gemini API with rate limiting"""
        can_request, reason = rate_limiter.can_make_request()
        if not can_request:
            logger.warning(f"Rate limit check failed: {reason}")
            return [0.0] * 768  # Return zero vector as fallback
        
        max_retries = 2  # Reduced retries
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_query"
                )
                rate_limiter.record_request()
                time.sleep(self.rate_limit_delay)
                return result['embedding']
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "quota" in error_msg.lower():
                    logger.error(f"API quota exceeded: {e}")
                    return [0.0] * 768  # Return zero vector immediately on quota error
                elif attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Embedding attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to embed query after {max_retries} attempts: {e}")
                    return [0.0] * 768

# Initialize embedding client
embeddings_client = GeminiEmbeddings()

# Initialize FastAPI app
app = FastAPI(title="Multi-Source RAG Medical Coding Chatbot with Gemini")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    chat_id: str
    user_id: str = "default_user"
    show_retrieved_data: bool = False

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    structured_query: Optional[Dict[str, Any]] = None
    retrieved_data: Optional[List[Dict[str, Any]]] = None
    warning: Optional[str] = None  # New field for warnings

def is_medical_query(question: str) -> bool:
    """Determine if the question is related to medical coding"""
    medical_keywords = [
        'icd', 'code', 'diagnosis', 'medical', 'condition', 'disease', 'symptom',
        'guideline', 'documentation', 'requirement', 'coding', 'clinical', 'health',
        'patient', 'treatment', 'procedure', 'assessment', 'record', 'chart', 'cpt'
    ]
    
    question_lower = question.lower()
    # Check for common greetings or general chat
    general_chat_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you']
    
    if any(pattern in question_lower for pattern in general_chat_patterns):
        return False
    
    return any(keyword in question_lower for keyword in medical_keywords)

def structure_user_input_fallback(question: str) -> Dict[str, Any]:
    """Fallback method for structuring user input without Gemini"""
    # Simple keyword extraction and structuring
    question_lower = question.lower()
    
    # Determine intent based on keywords
    intent = "Code Lookup"
    if any(word in question_lower for word in ['guideline', 'rule', 'how to']):
        intent = "Guideline Lookup"
    elif any(word in question_lower for word in ['include', 'exclude']):
        intent = "Inclusion/Exclusion Query"
    elif any(word in question_lower for word in ['vs', 'versus', 'compare', 'difference']):
        intent = "Comparison"
    
    # Extract potential codes (simple regex would be better)
    import re
    code_patterns = re.findall(r'[A-Z]\d{2}(?:\.\d+)?', question)
    
    return {
        "query": question,
        "intent": intent,
        "search_query": question,
        "filters": {
            "chapter": None,
            "section": None,
            "code": code_patterns[0] if code_patterns else None,
            "keywords": remove_stopwords(question),
            "patient": {"age": None, "gender": None},
            "include": [],
            "exclude": []
        }
    }

def structure_user_input_with_gemini(question: str) -> Dict[str, Any]:
    """Transform user query into structured format using Gemini with fallback"""
    # Check rate limits first
    can_request, reason = rate_limiter.can_make_request()
    if not can_request:
        logger.warning(f"Skipping Gemini structuring due to rate limits: {reason}")
        return structure_user_input_fallback(question)
    
    try:
        system_prompt = """You are an expert medical coding librarian. Return ONLY a JSON object:

{
  "query": "Rewritten search query",
  "intent": "Code Lookup|Guideline Lookup|Inclusion/Exclusion Query|Comparison|Clinical Scenario|Other",
  "search_query": "expanded query with medical terms",
  "filters": {
    "keywords": ["term1", "term2"],
    "code": "explicit code or null"
  }
}

Keep it concise. Return ONLY the JSON."""

        # Use Gemini Flash instead of Pro for better rate limits
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"{system_prompt}\n\nUser question: {question}"
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=512,
            )
        )
        
        rate_limiter.record_request()
        response_text = response.text
        
        # Try to parse JSON from response
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                structured_query = json.loads(json_str)
                return structured_query
        except json.JSONDecodeError:
            logger.warning("Failed to parse structured query JSON, using fallback")
        
        return structure_user_input_fallback(question)

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            logger.error(f"Gemini quota exceeded for structuring: {e}")
        else:
            logger.error(f"Error structuring user input with Gemini: {e}")
        return structure_user_input_fallback(question)

def search_unified_collection(structured_query: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
    """Search the unified Medical_Coder collection with filtering"""
    try:
        # Extract search terms
        search_query = structured_query.get("search_query", "")
        keywords = structured_query.get("filters", {}).get("keywords", [])
        
        # Combine search terms
        combined_query = f"{search_query} {' '.join(keywords)}" if keywords else search_query
        
        # Generate embedding
        query_embedding = embeddings_client.embed_query(combined_query)
        
        # Check if we got a valid embedding
        if all(x == 0.0 for x in query_embedding):
            logger.warning("Received zero vector embedding, search may be less accurate")
        
        # Build filters based on structured query
        filter_conditions = []
        
        # Filter by document group if we can infer it from the query
        if any(term in combined_query.lower() for term in ['guideline', 'rule', 'sequencing']):
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.doc_group",
                    match=models.MatchValue(value="ICD_CODES")
                )
            )
        elif any(term in combined_query.lower() for term in ['procedure', 'cpt']):
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.doc_group",
                    match=models.MatchValue(value="CPT_PROCEDURES")
                )
            )
        elif any(term in combined_query.lower() for term in ['terminology', 'definition']):
            filter_conditions.append(
                models.FieldCondition(
                    key="metadata.doc_group",
                    match=models.MatchValue(value="MEDICAL_TERMINOLOGY")
                )
            )
        
        # Create search filter
        search_filter = models.Filter(must=filter_conditions) if filter_conditions else None
        
        # Search the collection
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_embedding,
            limit=limit,
            query_filter=search_filter
        )
        
        # Format results
        results = []
        for result in search_results:
            metadata = result.payload.get("metadata", {})
            results.append({
                "text": result.payload.get("text", ""),
                "metadata": metadata,
                "score": result.score,
                "source_group": metadata.get("doc_group", "UNKNOWN"),
                "source_priority": metadata.get("group_priority", 99),
                "source_description": DOCUMENT_GROUPS.get(metadata.get("doc_group", ""), {}).get("description", "Unknown source"),
                "file_name": metadata.get("file_name", "Unknown"),
                "chunk_index": metadata.get("chunk_index", 0)
            })
        
        # Sort by priority and score
        results.sort(key=lambda x: (x["source_priority"], -x["score"]))
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching unified collection: {e}")
        return []

def generate_response_with_gemini(user_question: str, structured_query: Dict[str, Any], 
                                rag_results: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> tuple[str, Optional[str]]:
    """Generate response using Gemini with RAG data, returns (response, warning)"""
    # Check rate limits first
    can_request, reason = rate_limiter.can_make_request()
    if not can_request:
        warning = f"API quota limitations active: {reason}. Using fallback response."
        return generate_fallback_response(user_question, rag_results), warning
    
    try:
        # Prepare RAG content with source tags (limit content to avoid token limits)
        rag_content = ""
        source_list = []
        
        for i, result in enumerate(rag_results[:5]):  # Limit to top 5 results
            source_tag = f"[{result['source_group']}-{i+1}]"
            source_list.append(f"{source_tag}: {result['source_description']} (Score: {result['score']:.3f})")
            # Limit text length to avoid token limits
            text_snippet = result['text'][:800] + "..." if len(result['text']) > 800 else result['text']
            rag_content += f"\n{source_tag} {text_snippet}\n"
        
        system_prompt = f"""You are a medical coding assistant. Based on the retrieved information, provide a comprehensive answer.

User's question: {user_question}

Retrieved information:
{rag_content}

Provide a structured answer with relevant codes, guidelines, and clinical context. Keep response concise but comprehensive."""

        # Use Gemini Flash for better rate limits
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(
            system_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,  # Reduced token limit
            )
        )
        
        rate_limiter.record_request()
        return response.text, None

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            warning = "API quota exceeded. Using fallback response based on retrieved data."
            logger.error(f"Gemini quota exceeded for response generation: {e}")
            return generate_fallback_response(user_question, rag_results), warning
        else:
            logger.error(f"Error generating response with Gemini: {e}")
            return generate_fallback_response(user_question, rag_results), "API error occurred. Using fallback response."

def generate_fallback_response(user_question: str, rag_results: List[Dict[str, Any]]) -> str:
    """Generate a fallback response when Gemini is unavailable"""
    if not rag_results:
        return "I'm sorry, I couldn't retrieve relevant information for your question. Please try rephrasing your query or check if it's related to medical coding."
    
    # Create a simple response based on retrieved data
    response = f"Based on the retrieved medical coding information:\n\n"
    
    # Add top results
    for i, result in enumerate(rag_results[:3]):
        response += f"**Source {i+1} ({result['source_description']}):**\n"
        text_snippet = result['text'][:400] + "..." if len(result['text']) > 400 else result['text']
        response += f"{text_snippet}\n\n"
    
    response += "**Note:** This is a simplified response. For complete medical coding guidance, please consult the latest official coding manuals and guidelines."
    
    return response

def generate_general_response_with_gemini(question: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Generate response for non-medical queries using Gemini with fallback"""
    # Check rate limits first
    can_request, reason = rate_limiter.can_make_request()
    if not can_request:
        return "Hello! I'm a medical coding assistant specializing in ICD-10-CM and CPT coding. How can I help you with your medical coding questions today?"
    
    try:
        system_prompt = """You are a friendly medical coding assistant. Respond helpfully to general queries and mention your medical coding specialty when appropriate. Keep responses brief."""

        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content(
            f"{system_prompt}\n\nUser: {question}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=256,
            )
        )
        
        rate_limiter.record_request()
        return response.text

    except Exception as e:
        logger.error(f"Error generating general response: {e}")
        return "Hello! I'm here to help with your medical coding questions. How can I assist you today?"

async def get_conversation_history(chat_id: str, user_id: str, limit: int = 3):
    """Retrieve conversation history from Supabase with error handling"""
    try:
        # Try to import supabase
        try:
            from supabase import create_client, Client
        except ImportError:
            logger.warning("Supabase library not installed. Install with: pip install supabase")
            return []
        
        supabase_url = os.getenv("SUPABASE_URL", "https://ilnnwhsktxtuwhkcbaup.supabase.co")
        supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlsbm53aHNrdHh0dXdoa2NiYXVwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU4MDkwMDEsImV4cCI6MjA2MTM4NTAwMX0.tL6-RiUQJykGwzss_mZ5-LUB6XbqeTu4ihs89jd7OKs")
        supabase_table = os.getenv("SUPABASE_TABLE_NAME", "chathistory")
        
        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not configured")
            return []
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        response = supabase.table(supabase_table)\
            .select("*")\
            .eq("chat_id", chat_id)\
            .eq("user_id", user_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        if not response.data:
            return []
        
        history = []
        for msg in reversed(response.data):
            history.append({"role": "user", "content": msg["user_message"]})
            history.append({"role": "assistant", "content": msg["ai_message"]})
        
        return history
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        return []

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat API endpoint with better error handling"""
    try:
        # Get conversation history
        conversation_history = []
        if request.chat_id:
            conversation_history = await get_conversation_history(request.chat_id, request.user_id)
        
        # Check if it's a medical query
        if not is_medical_query(request.question):
            answer = generate_general_response_with_gemini(request.question, conversation_history)
            return ChatResponse(answer=answer, sources=[], structured_query=None)
        
        # Step 1: Structure the user input
        structured_query = structure_user_input_with_gemini(request.question)
        logger.info(f"Structured query: {structured_query}")
        
        # Step 2: Search unified collection
        rag_results = search_unified_collection(structured_query, limit=10)
        logger.info(f"Retrieved {len(rag_results)} results from unified collection")
        
        # Step 3: Generate response
        answer, warning = generate_response_with_gemini(request.question, structured_query, rag_results, conversation_history)
        
        # Prepare sources information
        sources = []
        for result in rag_results:
            sources.append({
                "file_name": result["file_name"],
                "source_group": result["source_group"],
                "source_description": result["source_description"],
                "source_priority": result["source_priority"],
                "score": result["score"],
                "chunk_index": result["chunk_index"]
            })
        
        # Prepare retrieved data if requested
        retrieved_data = None
        if request.show_retrieved_data:
            retrieved_data = []
            for i, result in enumerate(rag_results):
                retrieved_data.append({
                    "index": i,
                    "text": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"],
                    "full_text": result["text"],
                    "metadata": result["metadata"],
                    "score": result["score"],
                    "source_group": result["source_group"],
                    "file_name": result["file_name"],
                    "chunk_index": result["chunk_index"]
                })
        
        return ChatResponse(
            answer=answer, 
            sources=sources,
            structured_query=structured_query,
            retrieved_data=retrieved_data,
            warning=warning
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            answer="I'm sorry, I encountered an error while processing your request. Please try again.",
            sources=[],
            structured_query=None,
            warning="System error occurred"
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Qdrant connection
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == COLLECTION_NAME for col in collections.collections)
        
        # Test Gemini connection
        can_request, reason = rate_limiter.can_make_request()
        gemini_status = "available" if can_request else f"limited: {reason}"
        
        # Check Supabase availability
        supabase_status = "available"
        try:
            from supabase import create_client
            supabase_status = "installed"
        except ImportError:
            supabase_status = "not installed"
        
        return {
            "status": "healthy", 
            "collection_exists": collection_exists,
            "collection_name": COLLECTION_NAME,
            "groups": list(DOCUMENT_GROUPS.keys()),
            "gemini_status": gemini_status,
            "supabase_status": supabase_status,
            "qdrant_url": QDRANT_URL,
            "daily_requests_used": rate_limiter.daily_requests
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/api/collection/stats")
async def get_collection_stats():
    """Get statistics about the collection"""
    try:
        # Get total count
        total_count = qdrant_client.count(collection_name=COLLECTION_NAME).count
        
        stats = {"total_chunks": total_count, "groups": {}}
        
        # Get count by document group
        for group_name in DOCUMENT_GROUPS.keys():
            count = qdrant_client.count(
                collection_name=COLLECTION_NAME,
                count_filter=models.Filter(
                    must=[models.FieldCondition(
                        key="metadata.doc_group",
                        match=models.MatchValue(value=group_name)
                    )]
                )
            ).count
            stats["groups"][group_name] = count
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/api/search")
async def search_collection(query: str, limit: int = 5, group_filter: Optional[str] = None):
    """Direct search endpoint for testing"""
    try:
        structured_query = structure_user_input_with_gemini(query)
        results = search_unified_collection(structured_query, limit=limit)
        
        if group_filter:
            results = [r for r in results if r["source_group"] == group_filter]
        
        return {
            "query": query,
            "structured_query": structured_query,
            "results_count": len(results),
            "results": results
        }
        
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)
