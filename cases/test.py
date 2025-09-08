import ollama
import time

client = ollama.Client(host="127.0.0.1:11436")

messages = [
    {
        "role": "user",
        "content": "Why is the sky blue?"
    }
]

start_time = time.time()
response = client.chat(model="gemma3:4b",
                       messages=messages)

print(f"Response Time: {time.time() - start_time}")
print(response)
