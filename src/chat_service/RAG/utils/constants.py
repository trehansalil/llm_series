import os
import sys

from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

load_dotenv()

EMBEDDING_API_MODEL_URL=os.getenv("MODEL_API_URL_EMBEDDING", "")
EMBEDDING_API_MODEL_KEY=os.getenv("MODEL_API_KEY_EMBEDDING", "EMPTY")
EMBEDDING_API_MODEL_NAME=os.getenv("MODEL_API_NAME_EMBEDDING","bge-m3")

OPENAI_COMPATIBLE_API_KEY = os.getenv("MODEL_API_KEY_CHAT","")
OPENAI_COMPATIBLE_API_BASE = os.getenv("MODEL_API_URL_CHAT","")
OPENAI_COMPATIBLE_API_MODEL_NAME = os.getenv("MODEL_API_NAME_CHAT","")

EMBEDDING_MAX_LENGTH = int(os.getenv("MILVUS_EMBEDDING_MAX_LENGTH","1000"))
EMBEDDING_DIM = int(os.getenv("MILVUS_EMBEDDING_DIM","1024"))

# Sentry Settings
SENTRY_DSN = os.getenv("SENTRY_DSN","")

VERBOSE = 0