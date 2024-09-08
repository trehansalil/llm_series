import os
import random
import sys
import time
from typing import List

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import requests
import re
from src.chat_service.logging import logger
from src.chat_service.RAG.utils import log_time_to_sentry
from src.chat_service.RAG.utils.constants import (
    EMBEDDING_API_MODEL_URL, EMBEDDING_API_MODEL_KEY, OPENAI_COMPATIBLE_API_BASE, 
    OPENAI_COMPATIBLE_API_KEY, OPENAI_COMPATIBLE_API_MODEL_NAME, EMBEDDING_API_MODEL_NAME, VERBOSE
)


class LLMManager:

    system_message = """
        You are a Stock Market/Economic market expert, named `Sal Bot`, working with the Salil, the best Data Scientist of the Century. 
        Use positive language and avoid any criticism of the government or its policies.
        Do not mention yourself or start responses with phrases like 'Based on the provided context'. 
        Write answers directly as required, using phrases like 'According to my knowledge' when needed. 
        Bold/Italicize key points and use proper titles (eg. #, ##, ###) and bullet points without overdoing it.
    """

    def __init__(self) -> None:
        pass
    
    def clean_text(input_text: str) -> str:
        """
        Clean and format text for embedding processing
        
        Args:
        input_text (str): The input text Unicode characters.
        
        Returns:
        str: Cleaned and formatted text.
        """
        text = input_text.replace("\n", "<br/>")

        # Replace non-Language Unicode characters with their names
        text = re.sub(r'[^\x00-\x7F]', LLMManager.replace_non_Language_unicode, text)
        
        # Remove any remaining backslashes
        text = text.replace("\\", "")
        
        return text

    @staticmethod
    def inference_with_llm(
        prompt, 
        openai_key=OPENAI_COMPATIBLE_API_KEY, 
        openai_model=OPENAI_COMPATIBLE_API_MODEL_NAME, 
        max_tokens=5000, 
        temperature=0.2, 
        verbose=VERBOSE, 
        system_message="",
        **kwargs
    ):
        """Inference with inference_with_llm (OpenAI compatible API using /chat/completions endpoint deployed using Llama.cpp"""

        LLMManager.system_message = system_message or LLMManager.system_message

        return LLMManager.call(prompt, max_tokens, temperature,
                                        verbose, openai_key=openai_key,
                                        openai_model=openai_model, **kwargs)


    @staticmethod
    def call(prompt, max_tokens=256, temperature=0.2, verbose=VERBOSE,
             openai_key=OPENAI_COMPATIBLE_API_KEY, openai_model=OPENAI_COMPATIBLE_API_MODEL_NAME, **kwargs):
        headers = {
            'Authorization': f'Bearer {openai_key}',
            'Content-Type': 'application/json'
        }
                    
        payload = {
            "model": openai_model,
            "messages": [
                    {
                        "role": "system",
                        "content": LLMManager.system_message
                    },            
                    {
                        "role": "user",
                        "content": prompt
                    }
            ],
            "max_tokens": max_tokens,
            "stream": kwargs.get("stream", False),
            "temperature": temperature,
            "user": kwargs.get("user", "string"),
            "use_beam_search": kwargs.get("use_beam_search", False),
            "repetition_penalty": kwargs.get("repetition_penalty", 1),
            "length_penalty": kwargs.get("length_penalty", 1),
            "early_stopping": kwargs.get("early_stopping", False),
            "ignore_eos": kwargs.get("ignore_eos", False),
            "skip_special_tokens": kwargs.get("skip_special_tokens", True),
            "spaces_between_special_tokens": kwargs.get("spaces_between_special_tokens", True),
            "include_stop_str_in_output": kwargs.get("include_stop_str_in_output", False),
            "response_format": {
                "type": "text"
            }
        }
        try:
            response = requests.post(
                OPENAI_COMPATIBLE_API_BASE + "/v1/chat/completions",
                headers=headers, 
                json=payload,
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                chat = result["choices"][0]["message"]['content']
                
                finish_reason = result["choices"][0]["finish_reason"]
                if verbose:
                    print(f"Response: {chat}")
                    print(f"Finish reason: {finish_reason}")
                return chat
            elif response.status_code == 429:  # Too Many Requests
                if verbose:
                    print(f"Rate limit exceeded. Waiting before retry...")
                time.sleep(2 ** random.choice([3, 4, 5]))  # Exponential backoff
            else:
                print(f"Error Chat: {response.status_code} - {response.text}")
                logger.error(f"Error: {response.status_code} - {response.text}")
                return "", "error"
                      
        except requests.RequestException as e:
            print(f"Request failed: {e}. Retrying...")
            time.sleep(2 ** random.choice([3, 4, 5]))  # Exponential backoff     

    @log_time_to_sentry(step_name='LLMInferenceManager: generate_embedding')
    @staticmethod
    def generate_embedding(text, verbose: int) -> List:
        input_text = f'{text}'
        input_text = input_text.replace("\n", "<br/>")
        input_text = input_text.replace("§", "")
        input_text = input_text.replace("\\", " ")
        input_text = input_text.replace("\t", "------") 
        input_text = LLMManager.clean_text(input_text)

        if len(input_text) <= 15: # Embedding crashes if the length of text is smaller
            input_text = input_text + "<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>" # Add padding to the input text

        payload = {
            "model": EMBEDDING_API_MODEL_NAME,
            "input": input_text
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + EMBEDDING_API_MODEL_KEY
        }

        # print("Headers", headers)

        response = requests.post(EMBEDDING_API_MODEL_URL + "/embedding", json=payload, headers=headers, verify=False)
        if verbose:
            print(f"Embedding Input: \n{input_text}\n")
        if response.status_code == 200:
            embeddings = response.json()["data"][0]["embedding"]
            if verbose:
                print(f"\nEmbedding Model Response: \n{embeddings}\n")            
        else:
            print(f"Error: {response.status_code} - {response.text}")
            logger.error(f"Error LLM Inference: {response.status_code} - {response.text}")
            embeddings = []
            
        return embeddings