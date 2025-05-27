from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import ResponseHandlingException
import os
from dotenv import load_dotenv
import sys
import subprocess
import time

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

print(f"Connecting to Qdrant at: {QDRANT_URL}")

def check_docker_qdrant():
    """Check if Qdrant Docker container is running and fix if needed"""
    try:
        # Check if qdrant container exists
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=qdrant", "--format", "{{.Names}}:{{.Status}}"],
            capture_output=True, text=True, check=True
        )
        
        if "qdrant" in result.stdout:
            status = result.stdout.strip().split(':')[1] if ':' in result.stdout else ""
            print(f"Found Qdrant container with status: {status}")
            
            if "Up" not in status:
                print("Starting Qdrant container...")
                subprocess.run(["docker", "start", "qdrant"], check=True)
                time.sleep(5)  # Wait for container to start
            
            return True
        else:
            print("Qdrant container not found. Creating new one...")
            return create_qdrant_container()
            
    except subprocess.CalledProcessError:
        print("Docker not available or error checking containers")
        return False
    except Exception as e:
        print(f"Error checking Docker: {e}")
        return False

def create_qdrant_container():
    """Create and start a new Qdrant Docker container"""
    try:
        # Remove existing container if it exists
        subprocess.run(["docker", "rm", "-f", "qdrant"], capture_output=True)
        
        # Create data directory
        data_dir = os.path.expanduser("~/qdrant_data")
        os.makedirs(data_dir, exist_ok=True)
        print(f"Created data directory: {data_dir}")
        
        # Run new Qdrant container
        cmd = [
            "docker", "run", "-d",
            "--name", "qdrant",
            "-p", "6333:6333",
            "-p", "6334:6334",
            "-v", f"{data_dir}:/qdrant/storage",
            "qdrant/qdrant"
        ]
        
        print("Creating new Qdrant container...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Container created with ID: {result.stdout.strip()}")
        
        # Wait for container to be ready
        print("Waiting for Qdrant to start...")
        time.sleep(10)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to create Qdrant container: {e}")
        print("Please ensure Docker is installed and running")
        return False
    except Exception as e:
        print(f"Error creating container: {e}")
        return False

def test_connection_with_retry(max_retries=5):
    """Test connection to Qdrant with retries"""
    for attempt in range(max_retries):
        try:
            print(f"Connection attempt {attempt + 1}/{max_retries}...")
            
            # Use timeout and proper API key handling
            if QDRANT_API_KEY:
                qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=10)
            else:
                qdrant_client = QdrantClient(url=QDRANT_URL, timeout=10)
            
            # Test connection
            collections = qdrant_client.get_collections()
            print("✅ Connection successful!")
            return qdrant_client
            
        except Exception as e:
            print(f"❌ Connection attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 3 seconds...")
                time.sleep(3)
    
    return None

# First, try to fix Docker setup if connection fails
try:
    # Quick connection test
    if QDRANT_API_KEY:
        test_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5)
    else:
        test_client = QdrantClient(url=QDRANT_URL, timeout=5)
    test_client.get_collections()
    print("✅ Initial connection successful!")
    
except Exception as e:
    print(f"❌ Initial connection failed: {e}")
    print("Attempting to fix Qdrant setup...")
    
    if "localhost" in QDRANT_URL or "127.0.0.1" in QDRANT_URL:
        if check_docker_qdrant():
            print("✅ Qdrant container setup completed")
        else:
            print("❌ Failed to setup Qdrant container")
            print("\nManual setup instructions:")
            print("1. Install Docker if not installed")
            print("2. Run: docker pull qdrant/qdrant")
            print("3. Run: mkdir -p ~/qdrant_data")
            print("4. Run: docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v ~/qdrant_data:/qdrant/storage qdrant/qdrant")
            sys.exit(1)

# Now try to connect with retries
qdrant_client = test_connection_with_retry()

if not qdrant_client:
    print("❌ Failed to establish connection to Qdrant after multiple attempts")
    sys.exit(1)

# Collection configuration
collections = ["Medical_Coder"]
VECTOR_SIZE = 768  # Updated for Gemini embeddings (text-embedding-004)

for collection in collections:
    try:
        # Check if collection exists first
        if qdrant_client.collection_exists(collection):
            print(f"Collection '{collection}' already exists.")
            
            # Get collection info to check vector size
            try:
                collection_info = qdrant_client.get_collection(collection)
                current_size = collection_info.config.params.vectors.size
                
                if current_size != VECTOR_SIZE:
                    print(f"Collection has wrong vector size ({current_size} vs {VECTOR_SIZE}). Recreating...")
                    qdrant_client.delete_collection(collection)
                else:
                    print(f"Collection '{collection}' has correct configuration. Skipping creation.")
                    continue
                    
            except Exception as e:
                print(f"Error checking collection info: {e}. Recreating collection...")
                qdrant_client.delete_collection(collection)
        
        # Create new collection with correct vector size for Gemini
        print(f"Creating collection '{collection}' with vector size {VECTOR_SIZE}...")
        qdrant_client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        
        # Verify collection was created
        if qdrant_client.collection_exists(collection):
            print(f"✅ Collection '{collection}' was created successfully.")
            
            # Get collection stats
            collection_info = qdrant_client.get_collection(collection)
            print(f"   Vector size: {collection_info.config.params.vectors.size}")
            print(f"   Distance metric: {collection_info.config.params.vectors.distance}")
        else:
            print(f"❌ Collection '{collection}' creation verification failed.")
        
    except ResponseHandlingException as e:
        print(f"❌ Failed to create collection '{collection}': {e}")
        
        # Try to provide more specific error information
        if "directory" in str(e).lower():
            print("This appears to be a storage directory issue.")
            print("Try restarting your Qdrant container:")
            print("  docker restart qdrant")
            
    except Exception as e:
        print(f"❌ Unexpected error with collection '{collection}': {e}")

print("\n" + "="*50)
print("COLLECTION SETUP SUMMARY")
print("="*50)

try:
    collections_info = qdrant_client.get_collections()
    if collections_info.collections:
        for col in collections_info.collections:
            print(f"✅ Collection: {col.name}")
            try:
                details = qdrant_client.get_collection(col.name)
                print(f"   Points count: {details.points_count}")
                print(f"   Vector size: {details.config.params.vectors.size}")
                print(f"   Distance: {details.config.params.vectors.distance}")
            except Exception as e:
                print(f"   Error getting details: {e}")
            print()
    else:
        print("❌ No collections found")
except Exception as e:
    print(f"❌ Error getting collections summary: {e}")

print("Script completed.")
