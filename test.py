from src.chat_service.RAG.models.base import CustomOpenAI
from src.chat_service.RAG.utils.constants import OPENAI_COMPATIBLE_API_MODEL_NAME, OPENAI_COMPATIBLE_API_BASE, OPENAI_COMPATIBLE_API_KEY

print(OPENAI_COMPATIBLE_API_MODEL_NAME)

llm = CustomOpenAI(
    model=OPENAI_COMPATIBLE_API_MODEL_NAME,
    max_tokens=1000,
    api_key=OPENAI_COMPATIBLE_API_KEY,
    base_url=OPENAI_COMPATIBLE_API_BASE+'/v1'
)

llm.invoke("Write a 7-day itenary for Singapore")