import os
import tempfile
import tiktoken

MODEL = "gpt-4.1-mini"


print("=" * 70)
print("TIKTOKEN INFORMATION")
print("=" * 70)


# ---------------------------------------------------------
# Tiktoken installation
# ---------------------------------------------------------

print("\ntiktoken installed at:")
print(tiktoken.__file__)


# ---------------------------------------------------------
# Model -> encoding
# ---------------------------------------------------------

encoding = tiktoken.encoding_for_model(MODEL)

print("\nModel:")
print(MODEL)

print("\nEncoding:")
print(encoding.name)

print("\nVocabulary size:")
print(encoding.n_vocab)


# ---------------------------------------------------------
# Cache locations
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CACHE INFORMATION")
print("=" * 70)

custom_cache = os.environ.get("TIKTOKEN_CACHE_DIR")

print("\nTIKTOKEN_CACHE_DIR:")
print(custom_cache)

temp_directory = tempfile.gettempdir()

print("\nWindows TEMP directory:")
print(temp_directory)

default_cache = os.path.join(temp_directory, "data-gym-cache")

print("\nPossible default tiktoken cache:")
print(default_cache)

print("\nDoes it exist?")
print(os.path.exists(default_cache))


if os.path.exists(default_cache):

    print("\nCached files:")

    for filename in os.listdir(default_cache):

        full_path = os.path.join(default_cache, filename)

        print(filename, "size:", os.path.getsize(full_path), "bytes")


# ---------------------------------------------------------
# Actual token experiment
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOKEN EXPERIMENT")
print("=" * 70)

text = "ABC Electronics provides a 2-year warranty on laptops."

print("\nInput text:")
print(text)

token_ids = encoding.encode(text)

print("\nToken IDs:")
print(token_ids)

print("\nToken count:")
print(len(token_ids))


print("\n" + "-" * 70)

print(f"{'Position':<10}" f"{'Token ID':<15}" f"{'Token':<30}")

print("-" * 70)


for position, token_id in enumerate(token_ids):

    token_bytes = encoding.decode_single_token_bytes(token_id)

    token_text = token_bytes.decode("utf-8", errors="replace")

    print(f"{position:<10}" f"{token_id:<15}" f"{repr(token_text):<30}")


# ---------------------------------------------------------
# Specific word test
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("SPECIFIC WORD TEST")
print("=" * 70)

test_words = [
    "warranty",
    " warranty",
    "Warranty",
    " warranty.",
    "laptop",
    " laptop",
]

for word in test_words:

    ids = encoding.encode(word)

    print(repr(word), "->", ids)

# ---------------------------------------------------------
# First 100 token IDs
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 100 TOKEN IDs")
print("=" * 70)

for token_id in range(100):

    try:

        token_bytes = encoding.decode_single_token_bytes(token_id)

        token_text = token_bytes.decode("utf-8", errors="replace")

        print(token_id, repr(token_text))

    except Exception as error:

        print(token_id, "ERROR:", error)
