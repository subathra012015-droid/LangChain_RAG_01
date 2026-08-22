import tiktoken

model = "gpt-4.1-mini"

text = "ABC Electronics provides a 2-year warranty on laptops."

encoding = tiktoken.get_encoding("cl100k_base")

token_ids = encoding.encode(text)

print("Token IDs:", token_ids)

for token_id in token_ids:
    token = encoding.decode([token_id])
    print(token_id, repr(token))

print("Model:", model)
print("Tokenizer:", encoding.name)
print("Encoding name:", encoding.name)
print("Vocabulary size:", encoding.n_vocab)

for token_id in token_ids:
    token_bytes = encoding.decode_single_token_bytes(token_id)

    print(
        "Token ID:",
        token_id,
        "Token bytes:",
        token_bytes,
        "Text:",
        repr(token_bytes.decode("utf-8", errors="replace")),
    )
