import logging
import os
from pprint import pprint
from deepsearcher.offline_loading import load_from_local_files
from deepsearcher.configuration import Configuration, init_config
from deepsearcher.llm.openai_llm import OpenAI
from deepsearcher.online_query import query
httpx_logger = logging.getLogger("httpx")  # disable openai's logger output
httpx_logger.setLevel(logging.WARNING)

current_dir = os.path.dirname(os.path.abspath(__file__))

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-********"
# # Set your LangSmith API key and project name
os.environ["LANGSMITH_API_KEY"] = "************"
os.environ["LANGSMITH_PROJECT"] = "default" 

# Initialize configuration
config = Configuration()
config.llm_tracing = True # Enable tracing

init_config(config)

load_from_local_files(
    paths_or_directory=os.path.join(current_dir, "data/WhatisMilvus.pdf"),
    collection_name="milvus_docs",
    collection_description="All Milvus Documents",
    # force_new_collection=True, # If you want to drop origin collection and create a new collection every time, set force_new_collection to True
)

question = "Write a report comparing Milvus with other vector databases."

_, _, consumed_token = query(question, max_iter=1)
print(f"Consumed tokens: {consumed_token}")