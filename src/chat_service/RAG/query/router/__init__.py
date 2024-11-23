from concurrent.futures import ThreadPoolExecutor
import inspect
import time


from src.chat_service.RAG.models.base import RouterLLM
from src.chat_service.RAG.query.router.examples import *

from src.chat_service.RAG.query.router.models import StockNewsFinalModel
from src.chat_service.RAG.utils import log_response, log_time_to_sentry
from src.chat_service.RAG.utils.constants import VERBOSE

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import ChatPromptTemplate

class Params:
    def __init__(self, country: str):
        self.country = country


class GenericRouter:
    
    justify = []
    verbose = VERBOSE    
    
    def __init__(self, llm: RouterLLM, params = dict(country= "UAE")) -> None:
        """Initialize the router with a language model and embedding mechanism.

        Args:
            llm: A language model for processing queries.
        """

        self.llm: RouterLLM = llm
        self.chain = self.llm | JsonOutputParser()
        self.params = Params(
            country=params['country'],
        )
        
    @log_time_to_sentry(step_name="GenericSemanticRouter: __call__")
    def __call__(self, llm, query):
        """Process a query by routing it through a processing chain and print the results.

        Args:
            query: The user input query to process.
        """
        if self.stock_news_category == f'Stock_Market_News':
            template = ChatPromptTemplate.from_template(stock_news_template)
        else:
            template = ChatPromptTemplate.from_template(ooc_query_template)
        
        chain = (
            template 
            | llm
            | StrOutputParser() 
        )

        if GenericRouter.verbose:
            print(f"\nQuery: \n{query}\n")
        
        # response = chain.invoke(query)
        response = log_response(
            entity=chain, 
            prompt_value=query.strip(),
            step_name=f"Post Routing Chain Answering using `{self.__class__.__name__}` within {inspect.currentframe().f_code.co_name}"
        )              
        
        if GenericRouter.verbose:
            print(response)
        
        return { 
            "answer": response,
            "retrievals": [],
            "category": self.category,
            "sub_category": self.sub_category,
            "justify:": self.justify,
            "created": time.time(),
            "finish_reason": "complete"
        }

    def _capture_route_reasoning(self, response_instance):
        """Helper method to process the query using a template and return the category.

        Args:
            query: The user query to process.
            template: The template string for routing.

        Returns:
            An integer representing the chosen category.
        """
        
        self.justify.append(response_instance)
    
    def _other_classification(self, query):
        """
        Classify a news summary text into a category based on its content concurrently.

        Parameters:
        query (str): The news summary text to be classified.

        Returns:
        None: Sets self.category and self.sub_category based on the classification results.
        """
        self.justify = []
        
        with ThreadPoolExecutor() as executor:
            # Start all router function calls concurrently
            future_stock_news = executor.submit(self.stock_news_router, query)

            # Collect the results
            stock_news_response = future_stock_news.result()
        
        self.stock_news_category = stock_news_response['stock_news']['category']

    def stock_news_router(self, query):
        """Determine if a query relates to user messages type: Stock_Market_News or No_Stock_Market_News user messages.

            Args:
                query: The user query to classify.

            Returns:
                A string indicating the query category.
        """
        
        output = log_response(
            entity=self.chain, 
            prompt_value=query,
            response_model=StockNewsFinalModel,
            step_name=f"Routing Chain Classification using `{self.__class__.__name__}` within {inspect.currentframe().f_code.co_name}"
        )
        
        self._capture_route_reasoning(output['stock_news'])
        
        return output           

