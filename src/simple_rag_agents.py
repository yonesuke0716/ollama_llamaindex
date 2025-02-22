from dotenv import load_dotenv
from llama_index.llms.ollama import Ollama
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec

load_dotenv()


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and returns the product"""
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: float, b: float) -> float:
    """Add two numbers and returns the sum"""
    return a + b


add_tool = FunctionTool.from_defaults(fn=add)

# Model
llm = Ollama(model="llama3.1", request_timeout=120.0)

# RAG
documents = SimpleDirectoryReader("./data").load_data()
embed_model = OllamaEmbedding(model_name="nomic-embed-text")
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
query_engine = index.as_query_engine(llm=llm)

# Agents
# budget_tool = QueryEngineTool.from_defaults(
#     query_engine,
#     name="canadian_budget_2023",
#     description="A RAG engine with some basic facts about the 2023 Canadian federal budget.",
# )
# agent = ReActAgent.from_tools(
#     [multiply_tool, add_tool, budget_tool], llm=llm, verbose=True
# )

# # Response
# response = agent.chat(
#     "What is the total amount of the 2023 Canadian federal budget multiplied by 3? Go step by step, using a tool to do any math."
# )
# print(response)

finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply_tool, add_tool])
agent = ReActAgent.from_tools(finance_tools, llm=llm, verbose=True)

response = agent.chat("What is the current price of NVDA?")

print(response)
