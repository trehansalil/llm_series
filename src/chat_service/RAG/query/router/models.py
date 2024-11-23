from typing import Any, Dict, List, ClassVar, Tuple

from pydantic import BaseModel, Field

from src.chat_service.RAG.query.router.examples import *



class StockNewsModel(BaseModel):

    # # Define the list of example queries and their classifications
    schema: ClassVar[Dict[str, List[Tuple[str, str]]]] = {
        'ar': arabic_stock_market_news_examples,
        'en': english_stock_market_news_examples
    }
    
    @classmethod
    def get_schema(cls, language_code: str) -> List[Tuple[str, str]]:
        return cls.schema.get(language_code, [])  
    
    @classmethod
    def get_examples(cls, lang: str) -> str:
        return "\n".join([f"    - \"{query}\": {classification}" for query, classification in cls.get_schema(lang)])      

    # Define fields for the model
    
    choice: int = 1
    category: str = Field(
        description="""
        **Classification of the news text type, among the categories given below:**

        ### **1. Stock_Market_News**  
        Covers user messages containing information related to stock markets, financial updates, or trading activities.  
        Examples:  
        - Company earnings reports or forecasts  
        - Stock price movements and trends  
        - Market indices performance (e.g., S&P 500, NASDAQ)  
        - News about mergers, acquisitions, or IPOs  
        - Government policies impacting markets  
        - Interest rate changes or economic indicators  
        - Sector-specific financial developments (e.g., technology stocks, energy prices)  

        ### **2. No_Stock_Market_News**  
        Includes user messages unrelated to stock market or financial activities.  
        Examples:  
        - General non-financial news or events  
        - Personal financial questions unrelated to market movements  
        - Weather updates or unrelated global occurrences  
        - News about sports, entertainment, or lifestyle  
        - Questions about unrelated topics like technology or health  
        """,
        pattern=r'^(Stock_Market_News|No_Stock_Market_News)$',
        strict=True,
        example="No_Stock_Market_News"
    )
    chain_of_thought: str = Field(
        description="The chain of thought that led to the classification reason only.",
        example="Single data point that can be stated as text, which is categorized under No_Stock_Market_News."
    )

class StockNewsFinalModel(BaseModel):
    news_text: str = None
    stock_news: StockNewsModel
    
    @classmethod
    def set_definition(cls, lang: str) -> Any:
        """
        Dynamically updates the docstring for the class with example queries and classifications
        for `StockNewsModel`.
        """

        base = """
        A few-shot examples of text classification:

        Examples:
        """
        StockNewsModel.__doc__ = base
        StockNewsModel.__doc__ += StockNewsModel.get_examples(lang)
        
        return cls
