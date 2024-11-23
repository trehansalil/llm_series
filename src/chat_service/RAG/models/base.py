# Standard library imports
import inspect
import json
from operator import itemgetter
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

# Third-party library imports
import instructor
from instructor import Instructor
from openai import OpenAI
from pydantic import Field
from langchain.prompts import ChatPromptTemplate
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from langchain_core.messages import BaseMessage
from langchain_core.language_models import LanguageModelInput
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.callbacks import CallbackManagerForLLMRun, Callbacks
from langchain_core.outputs import Generation, LLMResult, RunInfo
from langchain_core.tools import BaseTool
from langchain_core.utils.pydantic import is_basemodel_subclass

# Internal module imports
from src.chat_service import LLMManager
from src.chat_service.logging import logger
from src.chat_service.RAG.models.embedding import CustomEmbeddings
from src.chat_service.RAG.utils import log_response, log_error
from src.chat_service.RAG.utils.constants import *

# Type checking (conditionally imported)
if TYPE_CHECKING:
    from langchain_core.pydantic_v1 import BaseModel
    from langchain_core.runnables import Runnable, RunnableConfig

# Type variables
_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[Dict[str, Any], Type[_BM]]
_DictOrPydantic = Union[Dict, _BM]





def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)  


class OpenAICompatibleLLM(BaseLLM):
    max_tokens: int = 200
    temperature: float = 0.2
    language: Optional[str] = "en"
    length_penalty: Optional[float] = 0.6
    top_p: Optional[float] = 1.0
    output_max_tokens: Optional[float] = 5000
    openai_api_base: str = OPENAI_COMPATIBLE_API_BASE
    openai_api_key: str = OPENAI_COMPATIBLE_API_KEY
    model: str = OPENAI_COMPATIBLE_API_MODEL_NAME
    retrievals: Optional[List[Any]] = []
    system_message: Optional[str] = ''
    response_format: Optional[str] = {"type": "text"}
    stream_: Optional[bool] = False
    
    def __init__(self, retrievals: Optional[List[Any]] = [], system_message: Optional[str] = '', **kwargs):
        super().__init__(**kwargs)
        self.retrievals = retrievals
        self.system_message=system_message or LLMManager.system_message
        self.openai_api_base = kwargs.get("openai_api_base", OPENAI_COMPATIBLE_API_BASE)
        self.openai_api_key = kwargs.get("openai_api_key", OPENAI_COMPATIBLE_API_KEY)
        self.model = kwargs.get("model", OPENAI_COMPATIBLE_API_MODEL_NAME)
        self.language = kwargs.get("language", "en")
        self.max_tokens = kwargs.get("max_tokens", None)
        self.output_max_tokens = kwargs.get("output_max_tokens", None)
        self.response_format = kwargs.get("response_format", {"type": "text"})
    
    def __call__(self, query: List[str], stop=None):
        return self._generate(prompts=[query], stop=stop)         
    
    def _check_for_mandatory_inputs(
        self, inputs: dict[str, Any], mandatory_params: List[str]
    ) -> bool:
        """Check for mandatory parameters in inputs"""
        for name in mandatory_params:
            if name not in inputs:
                logger.error(f"Mandatory input {name} missing from query")
                return False
        return True

    def _check_for_extra_inputs(
        self, inputs: dict[str, Any], all_params: List[str]
    ) -> bool:
        """Check for extra parameters not defined in the signature"""
        input_keys = set(inputs.keys())
        param_keys = set(all_params)
        if not input_keys.issubset(param_keys):
            extra_keys = input_keys - param_keys
            logger.error(
                f"Extra inputs provided that are not in the signature: {extra_keys}"
            )
            return False
        return True

    def _is_valid_inputs(
        self, inputs: List[Dict[str, Any]], function_schemas: List[Dict[str, Any]]
    ) -> bool:
        """Determine if the functions chosen by the LLM exist within the function_schemas,
        and if the input arguments are valid for those functions."""
        try:
            # Currently only supporting single functions for most LLMs in Dynamic Routes.
            if len(inputs) != 1:
                logger.error("Only one set of function inputs is allowed.")
                return False
            if len(function_schemas) != 1:
                logger.error("Only one function schema is allowed.")
                return False
            # Validate the inputs against the function schema
            if not self._validate_single_function_inputs(
                inputs[0], function_schemas[0]
            ):
                return False

            return True
        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            return False

    def _validate_single_function_inputs(
        self, inputs: Dict[str, Any], function_schema: Dict[str, Any]
    ) -> bool:
        """Validate the extracted inputs against the function schema"""
        try:
            # Extract parameter names and determine if they are optional
            signature = function_schema["signature"]
            param_info = [param.strip() for param in signature[1:-1].split(",")]
            mandatory_params = []
            all_params = []

            for info in param_info:
                parts = info.split("=")
                name_type_pair = parts[0].strip()
                if ":" in name_type_pair:
                    name, _ = name_type_pair.split(":")
                else:
                    name = name_type_pair
                all_params.append(name)

                # If there is no default value, it's a mandatory parameter
                if len(parts) == 1:
                    mandatory_params.append(name)

            # Check for mandatory parameters
            if not self._check_for_mandatory_inputs(inputs, mandatory_params):
                return False

            # Check for extra parameters not defined in the signature
            if not self._check_for_extra_inputs(inputs, all_params):
                return False

            return True
        except Exception as e:
            logger.error(f"Single input validation error: {str(e)}")
            return False

    def extract_function_inputs(
        self, query: str, function_schemas: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        logger.info("Extracting function input...")

        prompt = """
        You are an accurate and reliable computer program that only outputs valid JSON. 
        Your task is to output JSON representing the input arguments of a Python function.

        This is the Python function's schema:

        ### FUNCTION_SCHEMAS Start ###
            {function_schemas}
        ### FUNCTION_SCHEMAS End ###

        This is the input query.

        ### QUERY Start ###
            {query}
        ### QUERY End ###

        The arguments that you need to provide values for, together with their datatypes, are stated in "signature" in the FUNCTION_SCHEMAS.
        The values these arguments must take are made clear by the QUERY.
        Use the FUNCTION_SCHEMAS "description" too, as this might provide helpful clues about the arguments and their values.
        Return only JSON, stating the argument names and their corresponding values.

        ### FORMATTING_INSTRUCTIONS Start ###
            Return a respones in valid JSON format. Do not return any other explanation or text, just the JSON.
            The JSON-Keys are the names of the arguments, and JSON-values are the values those arguments should take.
        ### FORMATTING_INSTRUCTIONS End ###

        ### EXAMPLE Start ###
            === EXAMPLE_INPUT_QUERY Start ===
                "How is the weather in Hawaii right now in International units?"
            === EXAMPLE_INPUT_QUERY End ===
            === EXAMPLE_INPUT_SCHEMA Start ===
                {{
                    "name": "get_weather",
                    "description": "Useful to get the weather in a specific location",
                    "signature": "(location: str, degree: str) -> str",
                    "output": "<class 'str'>",
                }}
            === EXAMPLE_INPUT_QUERY End ===
            === EXAMPLE_OUTPUT Start ===
                {{
                    "location": "Hawaii",
                    "degree": "Celsius",
                }}
            === EXAMPLE_OUTPUT End ===
        ### EXAMPLE End ###

        Note: I will tip $500 for an accurate JSON output. You will be penalized for an inaccurate JSON output.

        Provide JSON output now:
        """
        
        response_prompt = ChatPromptTemplate.from_template(prompt)    
        
        chain = (
                    {
                        # Retrieve context using the normal question
                        "function_schemas": lambda x: x["function_schemas"],

                        # Pass on the question
                        "query": lambda x: x["query"],
                    }
                    | response_prompt
                    | self
                    | StrOutputParser()
                )  
        
        # output = chain.invoke({"query": query, "function_schemas": function_schemas})     
        output = log_response(
            entity=chain, 
            prompt_value={"query": query, "function_schemas": function_schemas},
            step_name=f"Functional calling using {self.__class__.__name__} within {inspect.currentframe().f_code.co_name}"
        )              
        
        # llm_input = [Message(role="user", content=prompt)]
        # llm_input = [prompt]
        # output = self(llm_input)
        if not output:
            raise Exception("No output generated for extract function input")
        output = output.replace("'", '"').strip().rstrip(",")
        logger.info(f"LLM output: {output}")
        function_inputs = json.loads(output)
        if not isinstance(function_inputs, list):
            function_inputs = [function_inputs]
        logger.info(f"Function inputs: {function_inputs}")
        if not self._is_valid_inputs(function_inputs, function_schemas):
            raise ValueError("Invalid inputs")
        return function_inputs             
    
    def bind_tools(
            self,
            tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
            *,
            tool_choice: Optional[
                Union[dict, str, Literal["auto", "none", "required", "any"], bool]
            ] = None,
            **kwargs: Any,
        ) -> Runnable[LanguageModelInput, BaseMessage]:
            """Bind tool-like objects to this chat model.

            Assumes model is compatible with OpenAI tool-calling API.

            Args:
                tools: A list of tool definitions to bind to this chat model.
                    Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                    models, callables, and BaseTools will be automatically converted to
                    their schema dictionary representation.
                tool_choice: Which tool to require the model to call.
                    Options are:
                    name of the tool (str): calls corresponding tool;
                    "auto": automatically selects a tool (including no tool);
                    "none": does not call a tool;
                    "any" or "required": force at least one tool to be called;
                    True: forces tool call (requires `tools` be length 1);
                    False: no effect;

                    or a dict of the form:
                    {"type": "function", "function": {"name": <<tool_name>>}}.
                **kwargs: Any additional parameters to pass to the
                    :class:`~langchain.runnable.Runnable` constructor.
            """

            formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
            if tool_choice:
                if isinstance(tool_choice, str):
                    # tool_choice is a tool/function name
                    if tool_choice not in ("auto", "none", "any", "required"):
                        tool_choice = {
                            "type": "function",
                            "function": {"name": tool_choice},
                        }
                    # 'any' is not natively supported by OpenAI API.
                    # We support 'any' since other models use this instead of 'required'.
                    if tool_choice == "any":
                        tool_choice = "required"
                elif isinstance(tool_choice, bool):
                    tool_choice = "required"
                elif isinstance(tool_choice, dict):
                    tool_names = [
                        formatted_tool["function"]["name"]
                        for formatted_tool in formatted_tools
                    ]
                    if not any(
                        tool_name == tool_choice["function"]["name"]
                        for tool_name in tool_names
                    ):
                        raise ValueError(
                            f"Tool choice {tool_choice} was specified, but the only "
                            f"provided tools were {tool_names}."
                        )
                else:
                    raise ValueError(
                        f"Unrecognized tool_choice type. Expected str, bool or dict. "
                        f"Received: {tool_choice}"
                    )
                kwargs["tool_choice"] = tool_choice
            return super().bind(tools=formatted_tools, **kwargs)    
      
    def with_structured_output(
            self,
            schema: Optional[_DictOrPydanticClass] = None,
            *,
            method: Literal["function_calling", "json_mode"] = "function_calling",
            include_raw: bool = False,
            **kwargs: Any,
        ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
            """Model wrapper that returns outputs formatted to match the given schema.

            Args:
                schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                    then the model output will be an object of that class. If a dict then
                    the model output will be a dict. With a Pydantic class the returned
                    attributes will be validated, whereas with a dict they will not be. If
                    `method` is "function_calling" and `schema` is a dict, then the dict
                    must match the OpenAI function-calling spec or be a valid JSON schema
                    with top level 'title' and 'description' keys specified.
                method: The method for steering model generation, either "function_calling"
                    or "json_mode". If "function_calling" then the schema will be converted
                    to an OpenAI function and the returned model will make use of the
                    function-calling API. If "json_mode" then OpenAI's JSON mode will be
                    used. Note that if using "json_mode" then you must include instructions
                    for formatting the output into the desired schema into the model call.
                include_raw: If False then only the parsed structured output is returned. If
                    an error occurs during model output parsing it will be raised. If True
                    then both the raw model response (a BaseMessage) and the parsed model
                    response will be returned. If an error occurs during output parsing it
                    will be caught and returned as well. The final output is always a dict
                    with keys "raw", "parsed", and "parsing_error".

            Returns:
                A Runnable that takes any ChatModel input and returns as output:

                    If include_raw is True then a dict with keys:
                        raw: BaseMessage
                        parsed: Optional[_DictOrPydantic]
                        parsing_error: Optional[BaseException]

                    If include_raw is False then just _DictOrPydantic is returned,
                    where _DictOrPydantic depends on the schema:

                    If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                        class.

                    If schema is a dict then _DictOrPydantic is a dict.

            Example: Function-calling, Pydantic schema (method="function_calling", include_raw=False):
                .. code-block:: python

                    from langchain_openai import ChatOpenAI
                    from langchain_core.pydantic_v1 import BaseModel


                    class AnswerWithJustification(BaseModel):
                        '''An answer to the user question along with justification for the answer.'''

                        answer: str
                        justification: str


                    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                    structured_llm = llm.with_structured_output(AnswerWithJustification)

                    structured_llm.invoke(
                        "What weighs more a pound of bricks or a pound of feathers"
                    )

                    # -> AnswerWithJustification(
                    #     answer='They weigh the same',
                    #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                    # )

            Example: Function-calling, Pydantic schema (method="function_calling", include_raw=True):
                .. code-block:: python

                    from langchain_openai import ChatOpenAI
                    from langchain_core.pydantic_v1 import BaseModel


                    class AnswerWithJustification(BaseModel):
                        '''An answer to the user question along with justification for the answer.'''

                        answer: str
                        justification: str


                    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                    structured_llm = llm.with_structured_output(
                        AnswerWithJustification, include_raw=True
                    )

                    structured_llm.invoke(
                        "What weighs more a pound of bricks or a pound of feathers"
                    )
                    # -> {
                    #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                    #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                    #     'parsing_error': None
                    # }

            Example: Function-calling, dict schema (method="function_calling", include_raw=False):
                .. code-block:: python

                    from langchain_openai import ChatOpenAI
                    from langchain_core.pydantic_v1 import BaseModel
                    from langchain_core.utils.function_calling import convert_to_openai_tool


                    class AnswerWithJustification(BaseModel):
                        '''An answer to the user question along with justification for the answer.'''

                        answer: str
                        justification: str


                    dict_schema = convert_to_openai_tool(AnswerWithJustification)
                    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                    structured_llm = llm.with_structured_output(dict_schema)

                    structured_llm.invoke(
                        "What weighs more a pound of bricks or a pound of feathers"
                    )
                    # -> {
                    #     'answer': 'They weigh the same',
                    #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                    # }

            Example: JSON mode, Pydantic schema (method="json_mode", include_raw=True):
                .. code-block::

                    from langchain_openai import ChatOpenAI
                    from langchain_core.pydantic_v1 import BaseModel

                    class AnswerWithJustification(BaseModel):
                        answer: str
                        justification: str

                    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
                    structured_llm = llm.with_structured_output(
                        AnswerWithJustification,
                        method="json_mode",
                        include_raw=True
                    )

                    structured_llm.invoke(
                        "Answer the following question. "
                        "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                        "What's heavier a pound of bricks or a pound of feathers?"
                    )
                    # -> {
                    #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
                    #     'parsed': AnswerWithJustification(answer='They are both the same weight.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'),
                    #     'parsing_error': None
                    # }

            Example: JSON mode, no schema (schema=None, method="json_mode", include_raw=True):
                .. code-block::

                    structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                    structured_llm.invoke(
                        "Answer the following question. "
                        "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                        "What's heavier a pound of bricks or a pound of feathers?"
                    )
                    # -> {
                    #     'raw': AIMessage(content='{\n    "answer": "They are both the same weight.",\n    "justification": "Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight." \n}'),
                    #     'parsed': {
                    #         'answer': 'They are both the same weight.',
                    #         'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The difference lies in the volume and density of the materials, not the weight.'
                    #     },
                    #     'parsing_error': None
                    # }


            """  # noqa: E501
            if kwargs:
                raise ValueError(f"Received unsupported arguments {kwargs}")
            is_pydantic_schema = _is_pydantic_class(schema)
            if method == "function_calling":
                if schema is None:
                    raise ValueError(
                        "schema must be specified when method is 'function_calling'. "
                        "Received None."
                    )
                tool_name = convert_to_openai_tool(schema)["function"]["name"]
                llm = self.bind_tools(
                    [schema], tool_choice=tool_name, parallel_tool_calls=False
                )
                if is_pydantic_schema:
                    output_parser: OutputParserLike = PydanticToolsParser(
                        tools=[schema],  # type: ignore[list-item]
                        first_tool_only=True,  # type: ignore[list-item]
                    )
                else:
                    output_parser = JsonOutputKeyToolsParser(
                        key_name=tool_name, first_tool_only=True
                    )
            elif method == "json_mode":
                llm = self.bind(response_format={"type": "json_object"})
                output_parser = (
                    PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
                    if is_pydantic_schema
                    else JsonOutputParser()
                )
            else:
                raise ValueError(
                    f"Unrecognized method argument. Expected one of 'function_calling' or "
                    f"'json_mode'. Received: '{method}'"
                )

            if include_raw:
                parser_assign = RunnablePassthrough.assign(
                    parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
                )
                parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
                parser_with_fallback = parser_assign.with_fallbacks(
                    [parser_none], exception_key="parsing_error"
                )
                return RunnableMap(raw=llm) | parser_with_fallback
            else:
                return llm | output_parser

    def _generate(self, prompts, stop=None, **kwargs):
        generations = []
        for prompt in prompts:

            token_count = CustomEmbeddings.count_tokens(prompt)
            system_token_count = CustomEmbeddings.count_system_tokens()
            if VERBOSE:
                logger.info(f"System Message Token Count: {system_token_count}")
            max_token = self.max_tokens
            if token_count + system_token_count > 4000:
                # Added the contribution coming from token count of the system message and prompt text to calculate the total token count
                max_token = self.output_max_tokens - token_count - system_token_count - 1000 # Adding some additional Offset here to avoid running into errors
            
            try:
                logger.info(f"Inferencing using {self.__class__.__name__} within {inspect.currentframe().f_code.co_name}\n")
                response = LLMManager().inference_with_llm(
                    prompt, 
                    temperature=kwargs.get('temperature', self.temperature),
                    top_p=kwargs.get('top_p', self.top_p),
                    max_tokens=kwargs.get('max_tokens', max_token),
                    length_penalty=kwargs.get('length_penalty', self.length_penalty),
                    openai_api_base=self.openai_api_base,
                    openai_api_key=self.openai_api_key,
                    model=self.model,
                    response_format = {"type": "text"} if self.response_format == "text" else self.response_format,
                    system_message=self.system_message,
                    language=self.language or "en",
                    stream=self.stream_
                )
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                logger.info(f"token count= {token_count}\nsystem token count = {system_token_count}")
                log_error(e=e, step_name=f"Error logged at `{self.__class__.__name__}` within {inspect.currentframe().f_code.co_name}")

            if self.stream_:
                generation = Generation(text="".join(response))
                generations.append(generation)
            else:
                val = ''.join(response)
                generation = Generation(text=val)
                generations.append(generation)

        return LLMResult(generations=[generations])

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "custom"    

class RouterLLM(BaseLLM):
    """
    A custom implementation of a Language Learning Model (LLM) Router
    to handle dynamic prompt generation and response formatting.

    Attributes:
        system_message (Optional[str]): Default message to guide the LLM's behavior.
        attributes (List[str]): List of attributes to include in the LLM response.
        remove_attributes (List[str]): Attributes to exclude from the LLM response.
        language (str): Language code (e.g., "en") for the LLM interaction.
        openai_api_base (str): API base URL for OpenAI.
        openai_api_key (str): API key for OpenAI.
        model (str): Name of the OpenAI model to use.
        top_p (float): Top-p sampling for the response generation.
        temperature (float): Temperature for controlling randomness in responses.
        max_tokens (int): Maximum number of tokens in a response.
        max_tries (int): Number of attempts for generating a valid response.
        frequency_penalty (float): Penalty to reduce repetitive text.
        presence_penalty (float): Penalty to reduce redundancy in context.
        response_format (Optional[Dict]): Format specification for the output.
        qualifier_dict (Optional[Dict]): Qualification rules for response attributes.
        complexity_dict (Optional[Dict]): Complexity rules for response attributes.
        input_model_params (Optional[Dict]): User-specified model parameters.
        model_params (Optional[Dict]): Parameters used during generation.
        client (Optional[Any]): OpenAI client instance for handling requests.
        set_params (List): List of parameters allowed to be directly set.

    Methods:
        __init__: Initializes the RouterLLM with optional parameters.
        _generate: Generates responses based on input prompts.
        _llm_type: Returns the LLM type as a string.
    """

    system_message: Optional[str] = Field(default=None)
    attributes: List[str] = []
    remove_attributes: List[str] = []
    language: str = Field(default="en")
    openai_api_base: str = LLAMA3_2_API_BASE
    openai_api_key: str = LLAMA3_2_API_KEY
    model: str = LLAMA3_2_API_MODEL_NAME
    top_p: float = 0.2
    temperature: float = 0
    max_tokens: int = 300
    max_tries: int = 1
    frequency_penalty: float = 0.5
    presence_penalty: float = 0.5
    response_format: Optional[Dict] = None
    stock_news_dict: Optional[Dict] = None
    input_model_params: Optional[Dict] = None
    model_params: Optional[Dict] = {}
    client: Optional[Any] = None
    set_params: List = ['remove_attributes', 'language', 'openai_api_base', 'openai_api_key', 'model']
    # , 'response_model'
    def __init__(self, system_message=None, **kwargs):
        """
        Initializes the RouterLLM instance with default or user-provided values.

        Args:
            system_message (str, optional): A message to guide the LLM's behavior.
            kwargs: Additional parameters for model configuration.
        """
        # Provide a default system message if none is supplied
        if system_message is None:
            system_message = "You are an expert at extracting information from user input. Always format your responses as JSON."

        # Pass system_message as part of kwargs for Pydantic validation
        kwargs['system_message'] = system_message

        # Initialize superclass attributes
        super().__init__(**kwargs)
        
        # Save the system message and other parameters
        self.system_message = kwargs['system_message']
        del kwargs['system_message']

        # Organize input parameters for further usage
        self.input_model_params = {}
        for key, value in kwargs.items():
            if '_dict' in key or key in self.set_params:
                setattr(self, key, value)
            elif key in ['tags', 'callbacks']:
                continue
            else:
                self.input_model_params[key] = value

        # Initialize OpenAI client with required settings
        self.client: Instructor = instructor.from_openai(
            OpenAI(
                base_url=self.openai_api_base,
                api_key=self.openai_api_key
            ),
            mode=instructor.Mode.TOOLS
        )
    
    def _generate(self, prompts: List[str], stop=None, **kwargs):
        """
        Generates a response for a list of prompts using the LLM client.

        Args:
            prompts (List[str]): A list of user input prompts for the LLM.
            stop (Optional[str]): Optional stopping criteria for the LLM generation.
            kwargs: Additional parameters for model behavior.

        Returns:
            LLMResult: The generated responses from the LLM.
        """
        # define the response model's language setup if the model exists
        if kwargs.get("response_model", None) is not None:
            response_model = kwargs.get("response_model").set_definition(self.language)
        else:
            raise f"`response_model` input not received at `{self.__class__.__name__}` within {inspect.currentframe().f_code.co_name}"
        
        generations = []
        for prompt in prompts:
            # Define parameters for generating a response
            self.model_params = dict(
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": prompt},
                ],
                response_model=response_model,
                model=self.model
            )
            self.model_params.update(self.input_model_params)

            try:
                # Generate response from OpenAI client
                response = self.client.chat.completions.create(**self.model_params)
            except Exception as e:
                print(f"Error while generating response: {e}")
                return {}

            # Set attributes to include in the response if not already defined
            if not self.attributes:
                self.attributes = list(response.__dict__.keys())
                for attribute in self.remove_attributes:
                    if attribute in self.attributes:
                        self.attributes.remove(attribute)

            response.news_text = prompt
            
            # Process response choices based on attributes and associated dictionaries
            for attribute in self.attributes:
                if hasattr(response, attribute) and hasattr(self, f"{attribute}_dict"):
                    try:
                        getattr(response, attribute).choice = getattr(self, f"{attribute}_dict")[getattr(response, attribute).category]
                    except Exception as e:
                        print(f"Response: {response} with erorr {e}")
                        raise e
            try:
                # Convert the response to JSON format and store it
                generation = Generation(text=response.model_dump_json(indent=2))
                generations.append(generation)
            except AttributeError as e:
                logger.error(f"Error generating response: {e}")
                log_error(e=e, step_name=f"Error logged at `{self.__class__.__name__}` within {inspect.currentframe().f_code.co_name}")

        return LLMResult(generations=[generations])

    def _generate_helper(
        self,
        prompts: list[str],
        stop: Optional[list[str]],
        run_managers: list[CallbackManagerForLLMRun],
        new_arg_supported: bool,
        **kwargs: Any,
    ) -> LLMResult:
        try:
            output = (
                self._generate(
                    prompts,
                    stop=stop,
                    # TODO: support multiple run managers
                    run_manager=run_managers[0] if run_managers else None,
                    **kwargs,
                )
                if new_arg_supported
                else self._generate(prompts, stop=stop, **kwargs)
            )
        except BaseException as e:
            for run_manager in run_managers:
                run_manager.on_llm_error(e, response=LLMResult(generations=[]))
            raise e
        flattened_outputs = output.flatten()
        for manager, flattened_output in zip(run_managers, flattened_outputs):
            manager.on_llm_end(flattened_output)
        if run_managers:
            output.run = [
                RunInfo(run_id=run_manager.run_id) for run_manager in run_managers
            ]
        return output

    @property
    def _llm_type(self):
        """
        Returns the type of the LLM being used.
        
        Returns:
            str: The type of the LLM, here 'router_llm'.
        """
        return "router_llm"

