from mlx_lm import load, generate
from mlx_lm.models.cache import make_prompt_cache 

model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

def create_prompt(text: str) -> str:
    prompt = text.strip()
    messages = [{"role": "user", "content": prompt}]

    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True)
    
    return prompt

# Create a prompt cache to store key-value pairs (conversation history) for the model
cache = make_prompt_cache(model)

prompt = create_prompt("what is a neutron star?")
text = generate(prompt=prompt, 
                model=model, 
                tokenizer=tokenizer, 
                prompt_cache= cache, 
                verbose=False)
print("response 1 = ", text)

prompt = create_prompt("how it's formed?")
text = generate(prompt=prompt, 
                model=model, 
                tokenizer=tokenizer, 
                prompt_cache= cache, 
                verbose=False)

print("response 2 = ", text)