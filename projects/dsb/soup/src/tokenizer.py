'''
This module provides the tokenizer for the code generation model.
'''
import tiktoken

def get_tokenizer():
    '''
    Returns a tiktoken tokenizer for the o200k_base encoding, extended with custom special tokens.

    The custom special tokens are:
    - <|startoftext|>
    - <|startofcode|>
    - <|endofcode|>

    Returns:
        tiktoken.Encoding: The configured tokenizer instance.
    '''
    # o200k_base tokenizer 로드
    o200k_base = tiktoken.get_encoding("o200k_base")

    # 특수 토큰 추가 (필요시)
    special_tokens = {
        "<|startoftext|>": 200018,
        "<|startofcode|>": 200019,
        "<|endofcode|>": 200020,
    }

    # extend the tokenizer with special tokens
    tokenizer = tiktoken.Encoding(
        name="o200k_with_special_tokens",
        pat_str=o200k_base._pat_str,
        mergeable_ranks=o200k_base._mergeable_ranks,
        special_tokens={**o200k_base._special_tokens, **special_tokens},
    )

    return tokenizer

if __name__ == '__main__':
    # Get the tokenizer
    tokenizer = get_tokenizer()

    # Example usage
    request = "Write a Python function to validate email addresses"
    print(f"Original request: {request}")

    # Encode the request
    tokens = tokenizer.encode(request)
    print(f"Encoded tokens: {tokens}")

    # Decode the tokens
    decoded = tokenizer.decode(tokens)
    print(f"Decoded request: {decoded}")

    # Example with special tokens
    code_snippet = '<|startofcode|>def hello():\n    print("Hello, world!")<|endofcode|>'
    print(f"\nOriginal code snippet: {code_snippet}")

    # Encode the code snippet
    code_tokens = tokenizer.encode(code_snippet, allowed_special={'<|startofcode|>', '<|endofcode|>'})
    print(f"Encoded code tokens: {code_tokens}")

    # Decode the code snippet
    decoded_code = tokenizer.decode(code_tokens)
    print(f"Decoded code snippet: {decoded_code}")