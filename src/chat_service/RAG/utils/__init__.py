from typing import Optional
import sentry_sdk
import time
from functools import wraps
from src.chat_service.RAG.utils.constants import ENVIRONMENT, SENTRY_DSN
from src.chat_service.logging import logger

sentry_sdk.init(
    dsn=SENTRY_DSN,  # Replace with your actual DSN
    send_default_pii=True, # send personally-identifiable information like LLM responses to sentry
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
)

def log_error(e, step_name: str, message: Optional[str]=None):
    """Logs the error from the LLM to sentry."""
    
    with sentry_sdk.start_transaction(op=step_name, name=f"The result of the {step_name.upper()}"):
        if message is None:
            sentry_sdk.capture_exception(e)
            logger.error(f"Error logging response at {step_name}: {e}")        
        else:
            sentry_sdk.capture_exception(e)
            logger.error(f"Error logging response at {step_name}: {e} and the message variable is {message}")                  

def log_response(entity, prompt_value, step_name: str, **kwargs):
    """Logs the response from the LLM to sentry."""
    with sentry_sdk.start_transaction(op=step_name, name=f"The result of the {step_name.upper()}"):
        try:
            # logger.info(f"Prompt Value: {prompt_value}")
            response = entity.invoke(prompt_value, **kwargs)
            logger.info(f"Response logged at {step_name}: {response}\n")
            return response
        except Exception as e:
            sentry_sdk.capture_exception(e)
            
            if ENVIRONMENT == 'development':
                log_error(e=e,  step_name=step_name, message=prompt_value)
                raise e
            else:
                log_error(e=e,  step_name=step_name, message=prompt_value)
                # logger.error(f"Error logging response at {step_name}: {e}")
                # logger.info(f"Prompt Value at {step_name}: {prompt_value}")
            
def log_time_to_sentry(step_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            sentry_sdk.capture_message(
                f"Step: {step_name}, Duration: {duration:.2f} seconds",
                level="info"
            )
            
            return result
        return wrapper
    return decorator