"""Simple script to display a star pyramid using a GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox


def build_pyramid(height: int) -> str:
    """Return the pyramid string for the given height."""
    lines = []
    for i in range(height):
        spaces = " " * (height - i - 1)
        stars = "*" * (2 * i + 1)
        lines.append(spaces + stars)
    return "\n".join(lines)


def generate(event: tk.Event | None = None) -> None:
    """Handle button press/Enter key to build the pyramid."""
    value = entry.get()
    try:
        n = int(value)
        if n <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter a positive integer.")
        return

    result_var.set(build_pyramid(n))


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Star Pyramid")

    frame = tk.Frame(root, padx=10, pady=10)
    frame.pack()

    entry_label = tk.Label(frame, text="Height:")
    entry_label.grid(row=0, column=0, sticky="e")

    entry = tk.Entry(frame)
    entry.grid(row=0, column=1)
    entry.bind("<Return>", generate)

    generate_btn = tk.Button(frame, text="Generate", command=generate)
    generate_btn.grid(row=0, column=2, padx=(5, 0))

    result_var = tk.StringVar()
    result_label = tk.Label(frame, textvariable=result_var, font=("Courier", 10), justify="left")
    result_label.grid(row=1, column=0, columnspan=3, pady=(10, 0))

    root.mainloop()
