def main(byte: int, target: int, base: int = 10):
    """
    Convert a number from one base to another.

    Positional arguments:
    byte  - The number to convert.
    target - The number to convert it to.
    base   - The base of the number. Defaults to 10.
    """
    _org = int(byte, base=base)
    if target == 2:
        print(bin(_org))
    elif target == 8:
        print(oct(_org))
    elif target == 10:
        print(_org)
    else:
        print(hex(_org))