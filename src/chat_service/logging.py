import logging

logging.basicConfig(level=logging.DEBUG)

# this is the main logger for the chat-api
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.debug("Logger initialized")
