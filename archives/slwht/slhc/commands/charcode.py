from __future__ import annotations

# def main(org: str | int) -> None:
def main(org) -> None:
    """
    Converts a character or an integer ASCII code to its corresponding representation.

    Args:
        org (str|int): A string or integer ASCII code to convert.

    Raises:
        TypeError: If the input is not a string or an integer.
    """
    if isinstance(org, int):
        print(chr(org))
    elif isinstance(org, str) and len(org) == 1:
        print(ord(org))
    else:
        raise TypeError("Input must be a single character string or an integer.")
