import warnings
import tiktoken
from src.chat_service import LLMManager
from src.chat_service.RAG.utils.constants import EMBEDDING_DIM, EMBEDDING_MAX_LENGTH, VERBOSE
from typing import List
from langchain_core.embeddings import Embeddings
from src.chat_service.RAG.utils import log_time_to_sentry
from src.chat_service.logging import logger

class CustomEmbeddings(Embeddings):
    """
    Custom Embeddings class for LangChain using a custom API.

    This class provides a custom implementation of the Embeddings class from LangChain,
    allowing you to use a custom API for generating embeddings.

    Attributes:
        score_threshold (float): The minimum score required for an embedding to be considered valid.
        embedding_dim (int): The dimensionality of the embeddings.
        max_length (int): The maximum length of the input text.

    Example:
        >>> custom_embeddings = CustomEmbeddings()
        >>> docs = ["This is a test document.", "This is another test document."]
        >>> embeddings = custom_embeddings(docs)
        >>> print(embeddings)
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    """
    score_threshold = 0.82
    tik = tiktoken.encoding_for_model("gpt-3.5-turbo")

    
    def __init__(self, **kwargs):
        """
        Initializes the CustomEmbeddings class with the API URL.

        Args:
            **kwargs: Additional keyword arguments.
        """
        self.embedding_dim = EMBEDDING_DIM
        self.max_length = EMBEDDING_MAX_LENGTH
        warnings.filterwarnings('ignore')
        super().__init__(**kwargs)   
        
    @log_time_to_sentry(step_name='CustomEmbeddings: __call__') 
    def __call__(self, docs: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents using the custom API.

        Args:
            docs (List[str]): A list of documents to be embedded.

        Returns:
            List[List[float]]: A list of embeddings for each document.

        Raises:
            ValueError: If the input list is empty or None.
        """        
        if (docs != None) & (docs != []):
            return self.embed_documents(texts=docs)
        else:
            raise ValueError("docs not found")
        
    def get_dim(self):
        """
        Returns the dimensionality of the embeddings.

        Returns:
            int: The dimensionality of the embeddings.
        """        
        self.embedding_dim = EMBEDDING_DIM
        return self.embedding_dim

    def get_length(self):
        """
        Returns the maximum length of the input text.

        Returns:
            int: The maximum length of the input text.
        """        
        self.max_length = EMBEDDING_MAX_LENGTH 
        return self.max_length         

    def embed_query(self, text: str, verbose: int = 0) -> List[float]:
        """
        Embeds a single query using the custom API.

        Args:
            text (str): The query to be embedded.

        Returns:
        List[float]: The embedding for the query.
        """

        embeddings = LLMManager.generate_embedding(text, verbose=verbose)
        return  embeddings

    @log_time_to_sentry(step_name='CustomEmbeddings: embed_documents')
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents using the custom API.

        Args:
            texts (List[str]): A list of documents to be embedded.

        Returns:
            List[List[float]]: A list of embeddings for each document.

        Example:
            >>> custom_embeddings = CustomEmbeddings()
            >>> docs = ["This is a test document.", "This is another test document."]
            >>> embeddings = custom_embeddings.embed_documents(docs)
            >>> print(embeddings)
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        """
        # with ThreadPoolExecutor() as executor:
        #     response_list = list(executor.map(lambda text: self.embed_query(text), texts))           
        response_list = [self.embed_query(text) for text in texts]
        # response.raise_for_status()  # Raise an exception for non-200 status codes
        return response_list

    @staticmethod
    def count_tokens(text):
        """
        Count the number of tokens in the input text.

        Args:
            text (str): The input text to count tokens for.

        Returns:
            int: The number of tokens in the input text.

        Example:
            >>> CustomEmbeddings.count_tokens("This is a sample sentence.")
            6
        """
        return len(CustomEmbeddings.tik.encode(text))
    
    @staticmethod
    def count_system_tokens():
        """
        Count the number of tokens in the system message.

        Args:
            text (str): The system message to count tokens for.

        Returns:
            int: The number of tokens in the system message.

        Example:
            >>> CustomEmbeddings.count_system_tokens(LLMManager.system_message)
            5
        """
        if VERBOSE:
            logger.warning(f"System Message: {LLMManager.system_message}")    
        return len(CustomEmbeddings.tik.encode(LLMManager.system_message))    
    
    @staticmethod
    def encode(text):
        """
        Encode the input text into tokens.

        Args:
            text (str): The input text to encode.

        Returns:
            list: A list of tokens representing the input text.

        Example:
            >>> CustomEmbeddings.encode("This is a sample sentence.")
            [1, 2, 3, 4, 5, 6]
        """
        return CustomEmbeddings.tik.encode(text)
    
    @staticmethod
    def decode(tokens):
        """
        Decode the input tokens back into text.

        Args:
            tokens (list): A list of tokens to decode.

        Returns:
            str: The decoded text.

        Example:
            >>> CustomEmbeddings.decode([1, 2, 3, 4, 5, 6])
            "This is a sample sentence."
        """
        return CustomEmbeddings.tik.decode(tokens)
