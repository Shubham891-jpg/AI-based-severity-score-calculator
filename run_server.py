#!/usr/bin/env python3
"""
IT Ticket Severity Calculator - Server Startup Script
"""

import os
import sys
import uvicorn
from pathlib import Path

def main():
    """Start the FastAPI server."""
    
    # Change to the project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Add the project directory to Python path
    sys.path.insert(0, str(project_dir))
    
    print("🚀 Starting IT Ticket Severity Calculator Server...")
    print("📍 Project Directory:", project_dir)
    print("🌐 Server will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔧 Admin Interface: http://localhost:8000/redoc")
    print("\n" + "="*60)
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    try:
        # Get port from environment variable (or default to universal standard 8080)
        port = int(os.environ.get("PORT", 8080))
        
        # Start the server
        uvicorn.run(
            "api.app:app",
            host="0.0.0.0",  # Bind to all interfaces
            port=port,
            reload=False,  # Set to True for development
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Ensure the model is trained: python src/models/train_model.py")
        print("3. Check if port 8000 is available")
        sys.exit(1)

if __name__ == "__main__":
    main()