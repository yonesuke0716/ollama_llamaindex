import ollama
import requests

available_functions = {
    "request": requests.request,
}

response = ollama.chat(
    "llama3.1",
    messages=[
        {
            "role": "user",
            "content": "get the ollama.com webpage?",
        }
    ],
    tools=[requests.request],
)

for tool in response.message.tool_calls or []:
    function_to_call = available_functions.get(tool.function.name)
    if function_to_call == requests.request:
        # Make an HTTP request to the URL specified in the tool call
        resp = function_to_call(
            method=tool.function.arguments.get("method"),
            url=tool.function.arguments.get("url"),
        )
        print(resp.text)
    else:
        print("Function not found:", tool.function.name)
