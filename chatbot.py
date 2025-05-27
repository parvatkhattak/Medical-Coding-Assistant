import os
import logging
from typing import List, Dict, Any, Optional
import json

from fastapi import FastAPI
import nltk
from pydantic import BaseModel
from qdrant_client import QdrantClient
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
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Document groups and their collection names
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

def get_gemini_embedding(text: str) -> List[float]:
    """Generate embedding using Gemini's text-embedding model"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        logger.error(f"Error generating Gemini embedding: {e}")
        # Fallback: return a zero vector of appropriate dimension (768 for text-embedding-004)
        return [0.0] * 768

def generate_gemini_response(messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """Generate response using Gemini Flash 2.0"""
    try:
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Convert messages to Gemini format
        # Gemini expects a conversation format, so we'll combine system and user messages
        system_message = ""
        conversation_parts = []
        
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            elif message["role"] == "user":
                conversation_parts.append(f"User: {message['content']}")
            elif message["role"] == "assistant":
                conversation_parts.append(f"Assistant: {message['content']}")
        
        # Combine system message with conversation
        full_prompt = f"{system_message}\n\n" + "\n\n".join(conversation_parts)
        
        # Generate response
        response = model.generate_content(
            full_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Error generating Gemini response: {e}")
        return "I'm sorry, I encountered an error while generating your response. Please try again."

# Initialize FastAPI app
app = FastAPI(title="Multi-Source RAG Medical Coding Chatbot")

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

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    structured_query: Optional[Dict[str, Any]] = None

def is_medical_query(question: str) -> bool:
    """Determine if the question is related to medical coding"""
    medical_keywords = [
        'icd', 'code', 'diagnosis', 'medical', 'condition', 'disease', 'symptom',
        'guideline', 'documentation', 'requirement', 'coding', 'clinical', 'health',
        'patient', 'treatment', 'procedure', 'assessment', 'record', 'chart'
    ]
    
    question_lower = question.lower()
    # Check for common greetings or general chat
    general_chat_patterns = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'how are you']
    
    if any(pattern in question_lower for pattern in general_chat_patterns):
        return False
    
    return any(keyword in question_lower for keyword in medical_keywords)

def structure_user_input(question: str) -> Dict[str, Any]:
    """Transform user query into structured format using chain-of-thought reasoning"""
    try:
        system_prompt = """You are an expert medical coding librarian and ICD-10–CM specialist. When given a user question about ICD-10 coding, follow this reasoning framework before producing output:

**STEP 1: Clarify Intent**
- "I observe the user's question: '{user_input}'."
- "Which category best fits?"
  • Code Lookup (find specific diagnosis code)
  • Guideline Lookup (rules—sequencing, code first)
  • Inclusion/Exclusion Query
  • Comparison (differences between codes)
  • Clinical Scenario (PDx vs SDx, E/M level)
  • Other
- "I conclude the intent is: ."

**STEP 2: Identify Key Entities & Context**
- "I note any explicit terms: body system/chapter, section/category, explicit codes."
- "I infer implicit context: patient age, gender, setting, severity, laterality, acute vs chronic."
- "Extracted metadata: {chapter}, {section}, {patient_context}, {qualifiers}."

**STEP 3: Enrich Query for Retrieval**
- "I expand abbreviations and add synonyms (e.g., 'MI' → 'myocardial infarction')."
- "I append relevant guideline references (e.g., 'use additional code for…')."
- "I include inclusion/exclusion terms based on official ICD-10–CM notes."

**STEP 4: Construct JSON Filters Object**
- "Now I assemble the final JSON with keys: `query`, `intent`, `search_query`, and `filters` (chapter, section, code, keywords, patient, include, exclude)."

**Deliverable:** Output **only** the JSON object in this format:

{
  "query": "Rewritten Search Query: ...",
  "intent": "Code Lookup",
  "search_query": "expanded natural-language query",
  "filters": {
    "chapter": "chapter name or null",
    "section": "section name or null", 
    "keywords": ["term1", "term2"],
    "patient": {"age": "adult/pediatric/null", "gender": "male/female/null"},
    "include": ["include term1"],
    "exclude": ["exclude term1"]
  }
}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"User question: {question}"}
        ]

        response_text = generate_gemini_response(messages, temperature=0.7, max_tokens=1024)
        
        # Try to parse JSON from response
        try:
            # Find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                structured_query = json.loads(json_str)
                return structured_query
        except json.JSONDecodeError:
            logger.warning("Failed to parse structured query JSON, using fallback")
        
        # Fallback structure
        return {
            "query": question,
            "intent": "Code Lookup",
            "search_query": question,
            "filters": {
                "chapter": None,
                "section": None,
                "code": None,
                "keywords": remove_stopwords(question),
                "patient": {"age": None, "gender": None},
                "include": [],
                "exclude": []
            }
        }

    except Exception as e:
        logger.error(f"Error structuring user input: {e}")
        # Return fallback structure
        return {
            "query": question,
            "intent": "Code Lookup", 
            "search_query": question,
            "filters": {"keywords": remove_stopwords(question)}
        }

def search_multi_source_rag(structured_query: Dict[str, Any], limit_per_source: int = 3) -> List[Dict[str, Any]]:
    """Search across multiple RAG sources with prioritization using Gemini embeddings"""
    all_results = []
    
    # Extract search terms
    search_query = structured_query.get("search_query", "")
    keywords = structured_query.get("filters", {}).get("keywords", [])
    
    # Combine search terms
    combined_query = f"{search_query} {' '.join(keywords)}" if keywords else search_query
    
    # Generate embedding using Gemini
    query_embedding = get_gemini_embedding(combined_query)
    
    # Search each group's collection
    for group_name, group_info in DOCUMENT_GROUPS.items():
        try:
            collection_name = group_info["collection"]
            
            # Search this collection
            search_result = qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=limit_per_source
            )
            
            # Add results with source information
            for result in search_result:
                all_results.append({
                    "text": result.payload.get("text", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "score": result.score,
                    "source_group": group_name,
                    "source_priority": group_info["priority"],
                    "source_description": group_info["description"]
                })
                
        except Exception as e:
            logger.warning(f"Error searching {group_name}: {e}")
            continue
    
    # Sort by priority (lower number = higher priority) then by score
    all_results.sort(key=lambda x: (x["source_priority"], -x["score"]))
    
    return all_results

def generate_rag_response(user_question: str, structured_query: Dict[str, Any], rag_results: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> str:
    """Generate response using chain-of-thought reasoning with RAG data"""
    try:
        # Prepare RAG content with source tags
        rag_content = ""
        source_list = []
        
        for i, result in enumerate(rag_results):
            source_tag = f"[{result['source_group']}-{i+1}]"
            source_list.append(f"{source_tag}: {result['source_description']}")
            rag_content += f"\n{source_tag} {result['text']}\n"
        
        system_prompt = """You are a certified ICD-10–CM coding assistant. You have three inputs:
1. User's original query: '{user_input}'
2. Retrieved RAG content (with source tags): {rag_results}
3. Source identifiers for each snippet: {source_list}

Before writing your answer, follow this chain-of-thought:

**STEP 1: Organize Retrieved Data**
- "I group the retrieved segments by source: GROUP1, GROUP2, GROUP3."
- "I note any overlapping or conflicting information."

**STEP 2: Apply Source Prioritization** 
- "I select answers from the highest-priority source available, in order: GROUP1 > GROUP2 > GROUP3."
- "If two sources at the same level conflict, I choose the most specific guideline language."

**STEP 3: Reason Through Code Selection**
- "I determine the Primary Diagnosis Code(s) that best match the clinical context."
- "I identify any Secondary Code(s) based on comorbidities, external causes, or additional factors."
-"If no specificity is provided, consider it as unspecified."

**STEP 4: Draft Response Outline**
- "I will list all relevant ICD-10–CM codes with descriptions."
- "I will provide a brief rationale (1–2 sentences) for each code, citing guideline notes or context."
- "If key information is missing, I formulate one concise clarification question."

**STEP 5: Finalize Structured Output**
- "I assemble the final formatted answer with codes, rationales, clarification (if needed), and a disclaimer."

Now produce the answer in the format below, and include **only** the specified sections (do not include your reasoning steps):

**ICD-10–CM Codes:**
- CODE – Description
- CODE – Description

**Rationale:**
- CODE: Explanation (reference guideline, includes/excludes, or context match)
- CODE: Explanation

**Clarification (if needed):**
[Your single question here]

**Disclaimer:** This answer is for informational purposes only. Please confirm with the latest ICD-10–CM guidelines or a certified medical coder."""

        user_prompt = f"""User's original query: {user_question}

Retrieved RAG content:
{rag_content}

Source identifiers:
{chr(10).join(source_list)}

Structured query context: {json.dumps(structured_query, indent=2)}"""

        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Last 3 exchanges
        
        messages.append({"role": "user", "content": user_prompt})

        return generate_gemini_response(messages, temperature=0.3, max_tokens=1024)

    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return "I'm sorry, I encountered an error while generating your answer. Please try again."

def generate_general_response(question: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Generate response for non-medical queries"""
    try:
        system_prompt = (
            "You are a friendly and professional medical coding assistant. "
            "For general queries and greetings, provide helpful and welcoming responses. "
            "If the conversation turns to medical coding topics, inform the user that you can help with ICD-10-CM coding questions."
        )

        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history[-4:])  # Last 2 exchanges
        
        messages.append({"role": "user", "content": question})

        return generate_gemini_response(messages, temperature=0.7, max_tokens=512)

    except Exception as e:
        logger.error(f"Error generating general response: {e}")
        return "Hello! I'm here to help with your ICD-10 coding questions. How can I assist you today?"

async def get_conversation_history(chat_id: str, user_id: str, limit: int = 3):
    """Retrieve conversation history from Supabase"""
    try:
        from supabase import create_client, Client
        
        supabase_url = os.getenv("SUPABASE_URL", "https://ilnnwhsktxtuwhkcbaup.supabase.co")
        supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlsbm53aHNrdHh0dXdoa2NiYXVwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU4MDkwMDEsImV4cCI6MjA2MTM4NTAwMX0.tL6-RiUQJykGwzss_mZ5-LUB6XbqeTu4ihs89jd7OKs")
        supabase_table = os.getenv("SUPABASE_TABLE_NAME", "chathistory")
        
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
    """Enhanced chat API endpoint with multi-source RAG pipeline"""
    try:
        # Get conversation history
        conversation_history = []
        if request.chat_id:
            conversation_history = await get_conversation_history(request.chat_id, request.user_id)
        
        # Check if it's a medical query
        if not is_medical_query(request.question):
            answer = generate_general_response(request.question, conversation_history)
            return ChatResponse(answer=answer, sources=[], structured_query=None)
        
        # Step 1: Structure the user input
        structured_query = structure_user_input(request.question)
        logger.info(f"Structured query: {structured_query}")
        
        # Step 2: Multi-source RAG retrieval
        rag_results = search_multi_source_rag(structured_query, limit_per_source=3)
        logger.info(f"Retrieved {len(rag_results)} results from RAG sources")
        
        # Step 3: Generate response with chain-of-thought reasoning
        answer = generate_rag_response(request.question, structured_query, rag_results, conversation_history)
        
        # Prepare sources information
        sources = []
        for result in rag_results:
            metadata = result["metadata"]
            sources.append({
                "file_name": metadata.get("file_name", "Unknown"),
                "source_group": result["source_group"],
                "source_description": result["source_description"],
                "source_priority": result["source_priority"],
                "score": result["score"]
            })
        
        return ChatResponse(
            answer=answer, 
            sources=sources,
            structured_query=structured_query
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            answer="I'm sorry, I encountered an error while processing your request. Please try again.",
            sources=[],
            structured_query=None
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "groups": list(DOCUMENT_GROUPS.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)
