import tkinter as tk
import sys
from io import StringIO

class DualConsoleRedirector(StringIO):
    def __init__(self, text_widget, original_stdout):
        super().__init__()
        self.text_widget = text_widget
        self.original_stdout = original_stdout

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.original_stdout.write(string)

    def flush(self):
        self.original_stdout.flush()

class ConsolePopup:
    def __init__(self, master):
        self.popup = None
        self.text_widget = None
        self.master = master
        self.original_stdout = sys.stdout


    def create_console_popup(self):
        if self.popup is None or not self.popup.winfo_exists():
            self.popup = tk.Toplevel(self.master)
            self.popup.title("Console Output")
            self.popup.attributes("-topmost", True)
            self.text_widget = tk.Text(self.popup, wrap=tk.WORD)
            self.text_widget.pack(expand=True, fill=tk.BOTH)

            scroll = tk.Scrollbar(self.popup)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)

            self.text_widget.config(yscrollcommand=scroll.set)
            scroll.config(command=self.text_widget.yview)

            # Redirect stdout to our custom stream
            sys.stdout = DualConsoleRedirector(self.text_widget, self.original_stdout)
        else:
            self.popup.lift()

def main():
    root = tk.Tk()
    root.title("Main Application")

    console_popup = ConsolePopup(root)

    open_console_button = tk.Button(root, text="Open Console", command=console_popup.create_console_popup)
    open_console_button.pack(pady=20)

    # Example output
    print_button = tk.Button(root, text="Print to Console", command=lambda: print("Hello from both consoles!"))
    print_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()