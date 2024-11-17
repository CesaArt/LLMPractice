from transformers import AutoModelForCausalLM, AutoTokenizer

# Load a pre-trained model and tokenizer
model_name = "openai-gpt"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to accommodate new token


# Generate text
prompt = "what is the purpose of life?"
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# Generate text with diversity
output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=100,
    do_sample=True,
    top_k=50,  # Sample from top 50 tokens
    top_p=0.8,  # Nucleus sampling
    temperature=1.2,  # Adds randomness
    pad_token_id=tokenizer.pad_token_id,
)

# Decode and print output
print(tokenizer.decode(output[0], skip_special_tokens=True))