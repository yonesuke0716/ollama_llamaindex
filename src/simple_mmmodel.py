from tomllib import loads
from llama_index.multi_modal_llms.ollama import OllamaMultiModal


with open("params.toml", encoding="utf-8") as fp:
    params = loads(fp.read())

mm_model = OllamaMultiModal(model=params["image_model"])