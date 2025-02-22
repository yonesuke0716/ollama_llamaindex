from pydantic import BaseModel, Field
import json
from tomllib import loads
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader


with open("params.toml", encoding="utf-8") as fp:
    params = loads(fp.read())


class Test(BaseModel):
    """写真を構成する要素の型定義"""

    title: str = Field(description="The name of title")
    summary: str = Field(description="The name of summary")


# Model
# llm = Ollama(model="llama3.1", request_timeout=120.0)
llm = Ollama(model=params["image_model"], images="./images/image01.jpg")

# RAG
documents = SimpleDirectoryReader("./data").load_data()
text = documents[0].text
sllm = llm.as_structured_llm(Test)

response = sllm.complete(text)
json_response = json.loads(response.text)
print(json.dumps(json_response, indent=2))
# embed_model = OllamaEmbedding(model_name="nomic-embed-text")
# index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
# query_engine = index.as_query_engine(llm=llm)

# # Agents
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
