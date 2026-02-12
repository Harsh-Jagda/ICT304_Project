import os


class Config:

    SECRET_KEY = "supersecretkey123"

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    USERS_FILE = os.path.join(BASE_DIR, "data", "users.csv")

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

    template_dir = os.path.join(BASE_DIR, "frontend", "templates")

