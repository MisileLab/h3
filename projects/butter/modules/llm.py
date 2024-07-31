from .config import api_key
from . import memory, llm_function

from langchain_openai import ChatOpenAI

functions = {**llm_function.functions, **memory.functions}
middle_converting_functions = {**llm_function.middle_converting_functions, **memory.middle_converting_functions}

llm = ChatOpenAI(model="gpt-4o", api_key=api_key)
llm = llm.bind_tools(list(functions.values()))
