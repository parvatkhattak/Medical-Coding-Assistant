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
    is_new_chat: bool = False  # Flag to indicate if this is a new chat session

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    structured_query: Optional[Dict[str, Any]] = None
    conversation_context: Optional[Dict[str, Any]] = None  # New field for context info

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

def is_follow_up_query(question: str, conversation_history: List[Dict[str, str]]) -> bool:
    """Determine if the current question is a follow-up to previous conversation"""
    if not conversation_history:
        return False
    
    follow_up_indicators = [
        'what about', 'how about', 'also', 'additionally', 'furthermore',
        'can you also', 'what if', 'in that case', 'regarding that',
        'about the previous', 'from the last', 'following up',
        'more about', 'expand on', 'clarify', 'explain further',
        'same patient', 'same case', 'this patient', 'this case'
    ]
    
    question_lower = question.lower()
    return any(indicator in question_lower for indicator in follow_up_indicators)

def extract_conversation_context(conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Extract relevant context from conversation history"""
    context = {
        "previous_codes": [],
        "patient_info": {},
        "topics_discussed": [],
        "last_query_intent": None
    }
    
    try:
        # Analyze the conversation to extract context
        for i, message in enumerate(conversation_history):
            if message["role"] == "user":
                # Extract potential medical codes mentioned
                import re
                code_patterns = [
                    r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b',  # ICD-10 codes
                    r'\b\d{5}(?:-\d{2})?\b'  # CPT codes
                ]
                for pattern in code_patterns:
                    codes = re.findall(pattern, message["content"])
                    context["previous_codes"].extend(codes)
                
                # Extract patient demographics/context
                if any(term in message["content"].lower() for term in ['patient', 'year old', 'male', 'female']):
                    age_match = re.search(r'(\d+)\s*year[s]?\s*old', message["content"].lower())
                    gender_match = re.search(r'\b(male|female|man|woman)\b', message["content"].lower())
                    
                    if age_match:
                        context["patient_info"]["age"] = age_match.group(1)
                    if gender_match:
                        context["patient_info"]["gender"] = gender_match.group(1)
            
            elif message["role"] == "assistant":
                # Extract topics from assistant responses
                if "ICD-10" in message["content"]:
                    context["topics_discussed"].append("ICD-10 coding")
                if "CPT" in message["content"]:
                    context["topics_discussed"].append("CPT procedures")
        
        # Remove duplicates
        context["previous_codes"] = list(set(context["previous_codes"]))
        context["topics_discussed"] = list(set(context["topics_discussed"]))
        
    except Exception as e:
        logger.error(f"Error extracting conversation context: {e}")
    
    return context

def structure_user_input_with_context(question: str, conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Transform user query into structured format using chain-of-thought reasoning with conversation context"""
    try:
        # Enhanced system prompt that considers conversation context
        system_prompt = """You are an expert medical coding librarian and ICD-10–CM specialist. When given a user question about ICD-10 coding, follow this reasoning framework before producing output:

**STEP 1: Analyze Conversation Context**
- Review any previous codes mentioned: {previous_codes}
- Consider patient information from context: {patient_info}
- Account for topics already discussed: {topics_discussed}
- Determine if this is a follow-up query or new topic

**STEP 2: Clarify Intent**
- "I observe the user's question: '{user_input}'."
- "Which category best fits?"
  • Code Lookup (find specific diagnosis code)
  • Guideline Lookup (rules—sequencing, code first)
  • Inclusion/Exclusion Query
  • Comparison (differences between codes)
  • Clinical Scenario (PDx vs SDx, E/M level)
  • Follow-up Query (building on previous discussion)
  • Other
- "I conclude the intent is: ."

**STEP 3: Identify Key Entities & Context**
- "I note any explicit terms: body system/chapter, section/category, explicit codes."
- "I infer implicit context: patient age, gender, setting, severity, laterality, acute vs chronic."
- "I consider previous conversation context for continuity."
- "Extracted metadata: {chapter}, {section}, {patient_context}, {qualifiers}."

**STEP 4: Enrich Query for Retrieval**
- "I expand abbreviations and add synonyms (e.g., 'MI' → 'myocardial infarction')."
- "I append relevant guideline references (e.g., 'use additional code for…')."
- "I include inclusion/exclusion terms based on official ICD-10–CM notes."
- "I incorporate context from previous queries if this is a follow-up."

**STEP 5: Construct JSON Filters Object**
- "Now I assemble the final JSON with keys: `query`, `intent`, `search_query`, `filters`, and `context_aware`."

**Deliverable:** Output **only** the JSON object in this format:

{{
  "query": "Rewritten Search Query: ...",
  "intent": "Code Lookup",
  "search_query": "expanded natural-language query",
  "context_aware": true/false,
  "filters": {{
    "chapter": "chapter name or null",
    "section": "section name or null", 
    "keywords": ["term1", "term2"],
    "patient": {{"age": "adult/pediatric/null", "gender": "male/female/null"}},
    "include": ["include term1"],
    "exclude": ["exclude term1"],
    "related_codes": ["previous codes if relevant"]
  }}
}}"""

        # Format the prompt with conversation context
        context_info = conversation_context or {}
        formatted_prompt = system_prompt.format(
            previous_codes=context_info.get("previous_codes", []),
            patient_info=context_info.get("patient_info", {}),
            topics_discussed=context_info.get("topics_discussed", [])
        )

        messages = [
            {"role": "system", "content": formatted_prompt},
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
        
        # Fallback structure with context awareness
        return {
            "query": question,
            "intent": "Code Lookup",
            "search_query": question,
            "context_aware": len(context_info.get("previous_codes", [])) > 0,
            "filters": {
                "chapter": None,
                "section": None,
                "code": None,
                "keywords": remove_stopwords(question),
                "patient": context_info.get("patient_info", {"age": None, "gender": None}),
                "include": [],
                "exclude": [],
                "related_codes": context_info.get("previous_codes", [])
            }
        }

    except Exception as e:
        logger.error(f"Error structuring user input: {e}")
        # Return fallback structure
        return {
            "query": question,
            "intent": "Code Lookup", 
            "search_query": question,
            "context_aware": False,
            "filters": {"keywords": remove_stopwords(question)}
        }

def search_multi_source_rag(structured_query: Dict[str, Any], limit_per_source: int = 3) -> List[Dict[str, Any]]:
    """Search across multiple RAG sources with prioritization using Gemini embeddings"""
    all_results = []
    
    # Extract search terms
    search_query = structured_query.get("search_query", "")
    keywords = structured_query.get("filters", {}).get("keywords", [])
    related_codes = structured_query.get("filters", {}).get("related_codes", [])
    
    # Combine search terms including related codes for context-aware search
    combined_query = f"{search_query} {' '.join(keywords)}"
    if related_codes:
        combined_query += f" {' '.join(related_codes)}"
    
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

def generate_rag_response_with_context(user_question: str, structured_query: Dict[str, Any], rag_results: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None, conversation_context: Dict[str, Any] = None) -> str:
    """Generate response using chain-of-thought reasoning with RAG data and full conversation context"""
    try:
        # Prepare RAG content with source tags
        rag_content = ""
        source_list = []
        
        for i, result in enumerate(rag_results):
            source_tag = f"[{result['source_group']}-{i+1}]"
            source_list.append(f"{source_tag}: {result['source_description']}")
            rag_content += f"\n{source_tag} {result['text']}\n"
        
        # Enhanced system prompt that considers full conversation context
        system_prompt = """You are a certified ICD-10–CM coding assistant with access to conversation history. You have these inputs:
1. User's current query: '{user_input}'
2. Retrieved RAG content (with source tags): {rag_results}
3. Source identifiers: {source_list}
4. Full conversation history for context
5. Conversation context: {conversation_context}

Before writing your answer, follow this enhanced chain-of-thought:

**STEP 1: Analyze Conversation Continuity**
- "I review the conversation history to understand the ongoing discussion."
- "I identify if this is a follow-up question or new topic."
- "I note any previous codes, patient details, or topics discussed."

**STEP 2: Organize Retrieved Data**
- "I group the retrieved segments by source: GROUP1, GROUP2, GROUP3."
- "I note any overlapping or conflicting information."
- "I consider how new information relates to previous discussion."

**STEP 3: Apply Source Prioritization** 
- "I select answers from the highest-priority source available, in order: GROUP1 > GROUP2 > GROUP3."
- "If two sources at the same level conflict, I choose the most specific guideline language."

**STEP 4: Reason Through Code Selection with Context**
- "I determine Primary Diagnosis Code(s) considering previous discussion context."
- "I identify Secondary Code(s) based on current query and conversation history."
- "I ensure consistency with previously provided information."
- "If no specificity is provided, consider it as unspecified."

**STEP 5: Draft Contextual Response**
- "I will acknowledge the conversation context when relevant."
- "I will list all relevant ICD-10–CM codes with descriptions."
- "I will provide rationale considering both current query and conversation flow."
- "If building on previous codes, I'll reference them appropriately."

**STEP 6: Finalize Structured Output**
- "I assemble the final formatted answer with contextual awareness."

Now produce the answer in the format below:

**Context Acknowledgment (if applicable):**
[Brief reference to previous discussion if this is a follow-up]

**ICD-10–CM Codes:**
- CODE – Description
- CODE – Description

**Rationale:**
- CODE: Explanation (reference guideline, includes/excludes, context match, or relationship to previous discussion)
- CODE: Explanation

**Clarification (if needed):**
[Your single question here]

**Disclaimer:** This answer is for informational purposes only. Please confirm with the latest ICD-10–CM guidelines or a certified medical coder."""

        user_prompt = f"""User's current query: {user_question}

Retrieved RAG content:
{rag_content}

Source identifiers:
{chr(10).join(source_list)}

Structured query context: {json.dumps(structured_query, indent=2)}

Conversation context: {json.dumps(conversation_context or {}, indent=2)}"""

        # Prepare messages with full conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add full conversation history for context-aware responses
        if conversation_history:
            # Include more of the conversation history for better context
            messages.extend(conversation_history[-10:])  # Last 5 exchanges instead of 3
        
        messages.append({"role": "user", "content": user_prompt})

        return generate_gemini_response(messages, temperature=0.3, max_tokens=1536)  # Increased token limit for more detailed responses

    except Exception as e:
        logger.error(f"Error generating RAG response: {e}")
        return "I'm sorry, I encountered an error while generating your answer. Please try again."

def generate_general_response(question: str, conversation_history: List[Dict[str, str]] = None) -> str:
    """Generate response for non-medical queries with conversation awareness"""
    try:
        system_prompt = (
            "You are a friendly and professional medical coding assistant. "
            "For general queries and greetings, provide helpful and welcoming responses. "
            "If the conversation turns to medical coding topics, inform the user that you can help with ICD-10-CM coding questions. "
            "Consider the conversation history to maintain continuity and provide contextually appropriate responses."
        )

        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
            messages.extend(conversation_history[-6:])  # Include more context for general chat
        
        messages.append({"role": "user", "content": question})

        return generate_gemini_response(messages, temperature=0.7, max_tokens=512)

    except Exception as e:
        logger.error(f"Error generating general response: {e}")
        return "Hello! I'm here to help with your ICD-10 coding questions. How can I assist you today?"

async def get_conversation_history(chat_id: str, user_id: str, limit: int = 50):  # Increased limit for full conversation
    """Retrieve full conversation history from Supabase"""
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

async def save_conversation_message(chat_id: str, user_id: str, user_message: str, ai_message: str):
    """Save conversation message to Supabase"""
    try:
        from supabase import create_client, Client
        
        supabase_url = os.getenv("SUPABASE_URL", "https://ilnnwhsktxtuwhkcbaup.supabase.co")
        supabase_key = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlsbm53aHNrdHh0dXdoa2NiYXVwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU4MDkwMDEsImV4cCI6MjA2MTM4NTAwMX0.tL6-RiUQJykGwzss_mZ5-LUB6XbqeTu4ihs89jd7OKs")
        supabase_table = os.getenv("SUPABASE_TABLE_NAME", "chathistory")
        
        supabase: Client = create_client(supabase_url, supabase_key)
        
        data = {
            "chat_id": chat_id,
            "user_id": user_id,
            "user_message": user_message,
            "ai_message": ai_message
        }
        
        supabase.table(supabase_table).insert(data).execute()
        
    except Exception as e:
        logger.error(f"Error saving conversation message: {e}")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat API endpoint with full conversation context awareness"""
    try:
        # Get full conversation history (unless it's explicitly a new chat)
        conversation_history = []
        if request.chat_id and not request.is_new_chat:
            conversation_history = await get_conversation_history(request.chat_id, request.user_id, limit=50)
        
        # Extract conversation context
        conversation_context = extract_conversation_context(conversation_history)
        
        # Determine if this is a follow-up query
        is_follow_up = is_follow_up_query(request.question, conversation_history)
        
        # Check if it's a medical query (consider follow-up context)
        is_medical = is_medical_query(request.question) or (is_follow_up and any("medical" in topic.lower() for topic in conversation_context.get("topics_discussed", [])))
        
        if not is_medical:
            answer = generate_general_response(request.question, conversation_history)
        else:
            # Step 1: Structure the user input with conversation context
            structured_query = structure_user_input_with_context(request.question, conversation_context)
            logger.info(f"Structured query: {structured_query}")
            
            # Step 2: Multi-source RAG retrieval
            rag_results = search_multi_source_rag(structured_query, limit_per_source=3)
            logger.info(f"Retrieved {len(rag_results)} results from RAG sources")
            
            # Step 3: Generate response with full conversation context
            answer = generate_rag_response_with_context(
                request.question, 
                structured_query, 
                rag_results, 
                conversation_history, 
                conversation_context
            )
        
        # Save the conversation message
        await save_conversation_message(request.chat_id, request.user_id, request.question, answer)
        
        # Prepare sources information
        sources = []
        if is_medical:
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
            structured_query=structured_query if is_medical else None,
            conversation_context={
                "is_follow_up": is_follow_up,
                "conversation_length": len(conversation_history),
                "context_extracted": conversation_context
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return ChatResponse(
            answer="I'm sorry, I encountered an error while processing your request. Please try again.",
            sources=[],
            structured_query=None,
            conversation_context=None
        )

@app.post("/api/new-chat")
async def create_new_chat():
    """Create a new chat session"""
    import uuid
    new_chat_id = str(uuid.uuid4())
    return {"chat_id": new_chat_id, "message": "New chat session created"}

@app.get("/api/chat-history/{chat_id}")
async def get_chat_history(chat_id: str, user_id: str = "default_user"):
    """Get conversation history for a specific chat"""
    try:
        history = await get_conversation_history(chat_id, user_id, limit=100)
        return {"chat_id": chat_id, "history": history}
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        return {"error": "Failed to retrieve chat history"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "groups": list(DOCUMENT_GROUPS.keys())}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbot:app", host="0.0.0.0", port=8000, reload=True)
