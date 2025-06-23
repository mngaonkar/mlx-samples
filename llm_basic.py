from mlx_lm import load, generate

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

prompt = "write a python code to drop pandas rows with missing values"
messages = [{"role": "user", "content": prompt}]

prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True)

text = generate(prompt=prompt, model=model, tokenizer=tokenizer, verbose=True)
print("response = ", text)