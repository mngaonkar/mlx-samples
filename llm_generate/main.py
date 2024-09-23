from mlx_lm import generate, load

checkpoint = "microsoft/phi-2"

def main():
    model, tokenizer = load(path_or_hf_repo=checkpoint)

    generation_args = {
        "temp": 0.5,
        "repetition_penalty": 1.2,
        "repetition_context_size": 20,
        "top_p": 0.95,
    }

    prompt = "write a python code to drop pandas rows with missing values"

    response = generate(
        model=model,
        tokenizer=tokenizer,
        max_tokens=200,
        prompt=prompt,
        verbose=True,
        **generation_args)
    
    print("response = ", response)

if __name__ == "__main__":
    main()