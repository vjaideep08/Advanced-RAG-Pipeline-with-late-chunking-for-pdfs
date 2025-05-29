# ==============================================================================
# UPDATED test_setup.py
# ==============================================================================
"""
Setup validation script for RAG Pipeline
Run this to check if your environment is properly configured
"""

import os
import sys
from pathlib import Path
import requests
import time

def check_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\n📦 Checking dependencies...")
    required_packages = [
        'fastapi', 'uvicorn', 'anthropic', 'transformers', 
        'torch', 'qdrant_client', 'PyPDF2', 'numpy', 'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').replace('_client', ''))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def check_environment():
    """Check environment variables"""
    print("\n🔧 Checking environment...")
    
    # Load .env file manually
    env_vars = {}
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value.strip('"').strip("'")
        
        if "CLAUDE_API_KEY" in env_vars:
            if env_vars["CLAUDE_API_KEY"]:
                print("✅ CLAUDE_API_KEY configured in .env")
            else:
                print("⚠️  CLAUDE_API_KEY is empty in .env")
        else:
            print("⚠️  CLAUDE_API_KEY not found in .env")
    else:
        print("⚠️  .env file not found")
    
    # Check Claude API key format
    claude_key = env_vars.get("CLAUDE_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if claude_key:
        if claude_key.startswith("sk-ant-"):
            print("✅ CLAUDE_API_KEY format looks correct")
        else:
            print("⚠️  CLAUDE_API_KEY format may be incorrect")
    else:
        print("⚠️  CLAUDE_API_KEY not set")
    
    return True

def check_pdf_directory():
    """Check PDF directory"""
    print("\n📁 Checking PDF directory...")
    pdf_path = os.getenv("PDF_PATH", "/home/imart/jayadeep/poc/rag_poc/pdfs/")
    pdf_dir = Path(pdf_path)
    
    if pdf_dir.exists():
        print(f"✅ PDF directory exists: {pdf_path}")
        pdf_files = list(pdf_dir.glob("*.pdf"))
        if pdf_files:
            print(f"✅ Found {len(pdf_files)} PDF files")
            for pdf in pdf_files[:3]:  # Show first 3
                print(f"   - {pdf.name}")
            if len(pdf_files) > 3:
                print(f"   ... and {len(pdf_files) - 3} more")
        else:
            print("⚠️  No PDF files found in directory")
    else:
        print(f"❌ PDF directory doesn't exist: {pdf_path}")
        print(f"   Create it with: mkdir -p {pdf_path}")
    
    return True

def check_qdrant():
    """Check if Qdrant is running"""
    print("\n🗄️  Checking Qdrant...")
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant is running")
            return True
        else:
            print(f"⚠️  Qdrant responded with status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("⚠️  Qdrant not running (will use in-memory fallback)")
    except Exception as e:
        print(f"⚠️  Error checking Qdrant: {e}")
    
    return True

def check_app_startup():
    """Check if the application can start"""
    print("\n🚀 Testing application startup...")
    print("   This will test imports and basic configuration...")
    
    try:
        # Create missing __init__.py files
        init_files = ['services/__init__.py', 'models/__init__.py', 'utils/__init__.py']
        for init_file in init_files:
            Path(init_file).touch(exist_ok=True)
        
        # Test imports
        from config import settings
        print("✅ Configuration loaded")
        
        from services.document_processor import DocumentProcessor
        print("✅ Document processor can be imported")
        
        from services.vector_store_factory import create_vector_store
        print("✅ Vector store factory can be imported")
        
        # Test basic initialization (without actually starting services)
        print("✅ Basic imports successful")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    return True

def create_sample_env():
    """Create a sample .env file"""
    print("\n📝 Creating sample .env file...")
    
    env_content = """# RAG Pipeline Configuration
CLAUDE_API_KEY=<Your claude api key here>

# Paths
PDF_PATH=/home/imart/jayadeep/poc/rag_poc/pdfs/

# Qdrant Configuration  
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=project_documents

# Optional: Override other settings
# EMBEDDING_MODEL=jinaai/jina-embeddings-v2-base-en
# MAX_CHUNK_LENGTH=1000
# TEMPERATURE=0.7
"""
    
    with open(".env.sample", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env.sample file")
    print("   Copy to .env and update with your actual values:")
    print("   cp .env.sample .env")
    print("   nano .env")

def main():
    """Run all checks"""
    print("🔍 RAG Pipeline Setup Validator")
    print("=" * 40)
    
    checks = [
        check_python_version,
        check_dependencies,
        check_environment,
        check_pdf_directory,
        check_qdrant,
        check_app_startup
    ]
    
    passed = 0
    for check in checks:
        if check():
            passed += 1
    
    print(f"\n📊 Summary: {passed}/{len(checks)} checks passed")
    
    if passed == len(checks):
        print("🎉 Your setup looks good! Try running:")
        print("   python main.py")
    else:
        print("⚠️  Some issues found. Check the messages above.")
        print("💡 Quick fixes:")
        print("   1. pip install -r requirements.txt")
        print("   2. Create .env file with CLAUDE_API_KEY")
        print("   3. Add PDF files to the PDF directory")
        print("   4. Start Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    
    # Offer to create sample .env
    if not Path(".env").exists():
        create_sample_env()

if __name__ == "__main__":
    main()
