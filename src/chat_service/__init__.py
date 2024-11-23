import json
import os
import random
import sys
import time
from typing import Dict, Generator, List, Optional
import unicodedata

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
import requests
import re
from src.chat_service.logging import logger
from src.chat_service.RAG.utils import log_error, log_time_to_sentry
from src.chat_service.RAG.utils.constants import (
    EMBEDDING_API_MODEL_URL, EMBEDDING_API_MODEL_KEY, OPENAI_COMPATIBLE_API_BASE, 
    OPENAI_COMPATIBLE_API_KEY, OPENAI_COMPATIBLE_API_MODEL_NAME, EMBEDDING_API_MODEL_NAME, VERBOSE, 
    _default_system_message
)


class LLMManager:

    system_message: str = _default_system_message
    lang_dict: Dict = {"ar": "Arabic", "en": "English"}

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
        verbose=VERBOSE, 
        system_message="",
        **kwargs
    ):
        """Inference with inference_with_llm (OpenAI compatible API using /chat/completions endpoint deployed using Llama.cpp"""

        language = kwargs.get("language", "en")
        
        LLMManager.system_message = system_message or LLMManager.system_message
        
        LLMManager.system_message = LLMManager.system_message.format(LLMManager.lang_dict[language])
        
        return LLMManager.call(prompt, verbose, **kwargs)


    @staticmethod
    def call(prompt, verbose=VERBOSE, **kwargs):
        headers = {
            'Authorization': f'Bearer {kwargs.get("openai_api_key", OPENAI_COMPATIBLE_API_KEY)}',
            'Content-Type': 'application/json'
        }
        
        logged_params = {
            "max_tokens": kwargs.get("max_tokens", 5000),
            "stream": kwargs.get("stream", False),
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 1),
            "user": kwargs.get("user", "string"),
            "use_beam_search": kwargs.get("use_beam_search", False),
            "presence_penalty": kwargs.get("presence_penalty", -1),
            "repetition_penalty": kwargs.get("repetition_penalty", 1),
            "length_penalty": kwargs.get("length_penalty", 1),
            "ignore_eos": kwargs.get("ignore_eos", False),
            "skip_special_tokens": kwargs.get("skip_special_tokens", True),
            "spaces_between_special_tokens": kwargs.get("spaces_between_special_tokens", True),
            "include_stop_str_in_output": kwargs.get("include_stop_str_in_output", False),
            "response_format": kwargs.get("response_format", {"type": "text"}),
            "openai_api_base": kwargs.get("openai_api_base", OPENAI_COMPATIBLE_API_BASE),
            "openai_api_key": kwargs.get("openai_api_key", OPENAI_COMPATIBLE_API_KEY),
            "model": kwargs.get("model", OPENAI_COMPATIBLE_API_MODEL_NAME),
            }
        print(f"\nLogged Param are as follows: {logged_params}\n")
                    
                 
        payload = {
            "model": kwargs.get("model", OPENAI_COMPATIBLE_API_MODEL_NAME),
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
            "max_tokens": kwargs.get("max_tokens", 5000),
            "stream": kwargs.get("stream", False),
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 1),
            "user": kwargs.get("user", "string"),
            "use_beam_search": kwargs.get("use_beam_search", False),
            "presence_penalty": kwargs.get("presence_penalty", -1),
            "repetition_penalty": kwargs.get("repetition_penalty", 1),
            "length_penalty": kwargs.get("length_penalty", 1),
            "ignore_eos": kwargs.get("ignore_eos", False),
            "skip_special_tokens": kwargs.get("skip_special_tokens", True),
            "spaces_between_special_tokens": kwargs.get("spaces_between_special_tokens", True),
            "include_stop_str_in_output": kwargs.get("include_stop_str_in_output", False),
            "response_format": kwargs.get("response_format", {"type": "text"})
        }

        try:
            response = requests.post(
                kwargs.get("openai_api_base", OPENAI_COMPATIBLE_API_BASE) + "/chat/completions",
                headers=headers, 
                json=payload,
                verify=True,
                stream=kwargs.get("stream", False)
            )

            if response.status_code == 200:
                if kwargs.get("stream", False):
                    # Directly yield from process_stream
                    for chunk in LLMManager.process_stream(response, verbose):
                        yield chunk
                else:
                    result = response.json()
                    chat = result["choices"][0]["message"]['content']
                    finish_reason = result["choices"][0]["finish_reason"]

                    if verbose:
                        print(f"Response: {chat}")
                        print(f"Finish reason: {finish_reason}")

                    yield str(f"{chat}")
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


    @staticmethod
    def process_stream(response, verbose: bool) -> Generator[str, None, None]:
        """
        Process a streaming response from the API.
        
        Args:
            response: The streaming response object
            verbose: Whether to print debug information
            
        Yields:
            Chunks of generated text as they arrive
        """
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                    
                if line.startswith(b"data: "):
                    json_str = line[6:].decode('utf-8')
                    
                    if json_str == "[DONE]":
                        break
                        
                    try:
                        json_data = json.loads(json_str)
                        if "choices" in json_data:
                            choice = json_data["choices"][0]
                            
                            if "delta" in choice and "content" in choice["delta"]:
                                content = choice["delta"]["content"]
                                if verbose:
                                    print(content, end='', flush=True)
                                yield content
                            
                            if verbose and "finish_reason" in choice and choice["finish_reason"]:
                                print(f"\nFinish reason: {choice['finish_reason']}")
                                
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON from stream: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
            yield f"Error in stream: {str(e)}"


    @log_time_to_sentry(step_name='LLMManager: generate_embedding')
    @staticmethod
    def generate_embedding(text, verbose: int, retries:Optional[int]=5) -> List:
        input_text = f'{text}'
        input_text = input_text.replace("\n", "<br/>")
        input_text = input_text.replace("§", "")
        input_text = input_text.replace('"', "\"")
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
        attempt = 0
        backoff_factor = 1  # Starting backoff factor for exponential increase
        
        while attempt < retries:
            try:
                response = requests.post(EMBEDDING_API_MODEL_URL, json=payload, headers=headers, verify=True)
                
                if verbose:
                    print(f"Embedding Input: \n{input_text}\n")
                
                if response.status_code == 200:
                    embeddings = response.json()["data"][0]["embedding"]
                    
                    if verbose:
                        print(f"\nEmbedding Model Response: \n{embeddings}\n")

                    return embeddings
                else:
                    raise Exception(f"Failed with status code {response.status_code} and message: {response.text}")

            except Exception as e:
                logger.error(f"Error Embedding model: {response.status_code} - {response.text}")
                
                attempt += 1
                if attempt < retries:
                    wait_time = backoff_factor * (2 ** attempt)  # Exponential backoff (2^attempt)
                    logger.info(f"Retrying in {wait_time} seconds... (Attempt {attempt}/{retries})")
                    time.sleep(wait_time)
                else:
                    logger.error("Max retries reached. Aborting.")
                    
                    # Raise an error after retries are exhausted
                    error_message = f"Max retries reached. Failed to get embeddings after {retries} attempts."
                    log_error(e=e, step_name="Embedding Generation", message=error_message)
                    logger.error(error_message)
                    raise Exception(error_message)