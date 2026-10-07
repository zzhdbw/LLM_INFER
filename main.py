from src.python import LLM, SamplingParams

llm = LLM("/home/models/Qwen/Qwen3-0.6B")
out = llm.generate(["你好呀"], SamplingParams(temperature=0, max_tokens=32))
print(out)
