"""Historical one-off sanity check retained for completeness (not exercised in CI)."""

import time

import ollama

client = ollama.Client(host="127.0.0.1:11436")
messages = [{"role": "user", "content": "Why is the sky blue?"}]
start = time.time()
response = client.chat(model="gemma3:4b", messages=messages)
print(f"Response time (s): {time.time() - start:.3f}")
print(response)
