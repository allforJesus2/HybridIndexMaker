import tkinter as tk
import sys
from io import StringIO
import time  # for testing


class DualConsoleRedirector(StringIO):
    def __init__(self, text_widget, original_stdout):
        super().__init__()
        self.text_widget = text_widget
        self.original_stdout = original_stdout

    def write(self, string):
        try:
            if self.text_widget and self.text_widget.winfo_exists():
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
                self.text_widget.update_idletasks()  # Force update of the widget
            self.original_stdout.write(string)
            self.flush()  # Ensure immediate flushing
        except tk.TclError:
            # If widget is destroyed, just write to original stdout
            self.original_stdout.write(string)
            self.flush()

    def flush(self):
        self.original_stdout.flush()
        try:
            if self.text_widget and self.text_widget.winfo_exists():
                self.text_widget.update()  # Force GUI update
        except tk.TclError:
            pass


class ConsolePopup:
    def __init__(self, master):
        self.popup = None
        self.text_widget = None
        self.master = master
        self.original_stdout = sys.stdout
        self.redirector = None

    def create_console_popup(self):
        if self.popup is None or not self.popup.winfo_exists():
            self.popup = tk.Toplevel(self.master)
            self.popup.title("Console Output")
            self.popup.attributes("-topmost", True)

            # Set a reasonable size for the popup
            self.popup.geometry("600x400")

            # Configure the popup background
            self.popup.configure(bg='black')

            # Bind the window close event
            self.popup.protocol("WM_DELETE_WINDOW", self.on_closing)

            # Create and configure the text widget with colors
            self.text_widget = tk.Text(
                self.popup,
                wrap=tk.WORD,
                bg='black',  # Background color
                fg='#00ff00',  # Text color (bright green)
                insertbackground='#00ff00',  # Cursor color
                selectbackground='#005500',  # Selection background
                selectforeground='#00ff00',  # Selection text color
                font=('Courier', 10)  # Monospace font
            )
            self.text_widget.pack(expand=True, fill=tk.BOTH)

            # Create and configure the scrollbar
            scroll = tk.Scrollbar(self.popup, bg='black', troughcolor='black')
            scroll.pack(side=tk.RIGHT, fill=tk.Y)

            self.text_widget.config(yscrollcommand=scroll.set)
            scroll.config(command=self.text_widget.yview)

            # Create new redirector and set stdout
            self.redirector = DualConsoleRedirector(self.text_widget, self.original_stdout)
            sys.stdout = self.redirector
        else:
            self.popup.lift()

    def on_closing(self):
        # Restore original stdout before destroying the window
        sys.stdout = self.original_stdout
        self.popup.destroy()
        self.popup = None
        self.text_widget = None
        self.redirector = None


def test_loop():
    # Example of a long-running process with prints
    for i in range(5):
        print(f"Processing item {i}...")
        time.sleep(1)  # Simulate some work
    print("Processing complete!")


def main():
    root = tk.Tk()
    root.title("Main Application")

    console_popup = ConsolePopup(root)

    open_console_button = tk.Button(root, text="Open Console",
                                    command=console_popup.create_console_popup)
    open_console_button.pack(pady=20)

    # Example output buttons
    print_button = tk.Button(root, text="Print to Console",
                             command=lambda: print("Hello from both consoles!"))
    print_button.pack(pady=10)

    # Add a test loop button
    loop_button = tk.Button(root, text="Run Test Loop",
                            command=test_loop)
    loop_button.pack(pady=10)

    root.mainloop()


if __name__ == "__main__":
    main()