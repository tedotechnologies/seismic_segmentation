from dotenv import load_dotenv
import os

load_dotenv()

CLEARML_WEB_HOST = os.getenv("CLEARML_WEB_HOST")
CLEARML_API_HOST = os.getenv("CLEARML_API_HOST")
CLEARML_FILES_HOST = os.getenv("CLEARML_FILES_HOST")
CLEARML_API_ACCESS_KEY = os.getenv("CLEARML_API_ACCESS_KEY")
CLEARML_API_SECRET_KEY = os.getenv("CLEARML_API_SECRET_KEY")
