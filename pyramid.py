"""Simple script to print a star pyramid."""


def star_pyramid(height: int) -> None:
    """Print a star pyramid of the given height."""
    for i in range(height):
        spaces = ' ' * (height - i - 1)
        stars = '*' * (2 * i + 1)
        print(spaces + stars)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("Please provide a valid integer height.")
            sys.exit(1)
    else:
        n = int(input("Enter pyramid height: "))

    if n <= 0:
        print("Height must be a positive integer.")
    else:
        star_pyramid(n)
