import os

class Config:
    SECRET_KEY = "your-secret-key-here-change-in-production"
    
    # Base directory is backend/ folder
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Project root is one level up from backend/
    PROJECT_ROOT = os.path.dirname(BACKEND_DIR)
    
    # Templates are in ICT304_Project/frontend/templates/
    template_dir = os.path.join(PROJECT_ROOT, "frontend", "templates")
    
    # Uploads folder in backend/uploads/
    UPLOAD_FOLDER = os.path.join(BACKEND_DIR, "uploads")
    
    # Demo data is in backend/data/
    DEMO_DATA_PATH = os.path.join(BACKEND_DIR, "data", "processed_data_ca.parquet")
    
    # users file
    USERS_FILE = os.path.join(BACKEND_DIR, "users.csv")
    
    # Create folders if they don't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Debug output
    print(f"CONFIG DEBUG:")
    print(f"   BACKEND_DIR: {BACKEND_DIR}")
    print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"   template_dir: {template_dir}")
    print(f"   UPLOAD_FOLDER: {UPLOAD_FOLDER}")
    print(f"   DEMO_DATA_PATH: {DEMO_DATA_PATH}")
    print(f"   USERS_FILE: {USERS_FILE}")
