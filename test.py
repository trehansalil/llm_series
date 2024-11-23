from src.chat_service.RAG.models.base import OpenAICompatibleLLM
from src.chat_service.RAG.utils.constants import OPENAI_COMPATIBLE_API_MODEL_NAME

print(OPENAI_COMPATIBLE_API_MODEL_NAME)

llm = OpenAICompatibleLLM(
    max_tokens=1000,
)

print(llm.invoke("Write a 7-day itenary for Singapore"))