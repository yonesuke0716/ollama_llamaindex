import asyncio
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    ReActAgent,
)
from llama_index.core.tools import FunctionTool


# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b


llm = Ollama(model="llama3.1", request_timeout=120.0)
Settings.llm = llm
# Create agent configs
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant.",
    tools=[
        FunctionTool.from_defaults(fn=add),
        FunctionTool.from_defaults(fn=subtract),
    ],
    lm=llm,
)

retriever_agent = ReActAgent(
    name="retriever",
    description="Manages data retrieval",
    system_prompt="You are a retrieval assistant.",
    lm=llm,
)

# Create and run the workflow
workflow = AgentWorkflow(
    agents=[calculator_agent, retriever_agent], root_agent="calculator"
)


async def main():
    # Run the system
    response = await workflow.run(user_msg="Can you add 5 and 3?")
    print(response)

    # Or stream the events
    handler = workflow.run(user_msg="Can you add 5 and 3?")
    async for event in handler.stream_events():
        if hasattr(event, "delta"):
            print(event.delta, end="", flush=True)


# Run the async function
asyncio.run(main())
