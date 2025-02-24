import tkinter as tk
from PIL import Image, ImageTk

class SplashScreen:
    def __init__(self, parent, image_path):
        self.parent = parent

        # Create a toplevel window
        self.splash = tk.Toplevel(parent)
        self.splash.overrideredirect(True)  # Remove window decorations

        # Configure the window to handle transparency
        self.splash.attributes('-alpha', 1.0)  # Make the window fully opaque
        self.splash.wm_attributes('-transparentcolor', 'black')  # Set transparent color

        # Load the image preserving transparency
        image = Image.open(image_path)
        # Convert to RGBA if it isn't already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        splash_width, splash_height = image.size

        # Create a fully transparent background
        background = Image.new('RGBA', image.size, (0, 0, 0, 0))
        # Composite the image onto the transparent background
        background.paste(image, (0, 0), image)

        # Resize if needed
        background = background.resize((splash_width, splash_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(background)

        # Create and pack the image label with transparent background
        self.label = tk.Label(self.splash, image=self.photo, border=0, bg='black')
        self.label.pack()

        # Add a loading label with transparent background
        self.loading_label = tk.Label(
            self.splash,
            text="Loading...",
            font=("Arial", 12),
            bg='black',  # Match the transparent color
            fg='white'  # Make text visible
        )
        self.loading_label.pack(pady=10)

        # Configure the splash window background
        self.splash.configure(bg='black')  # Match the transparent color

        # Center the splash screen
        screen_width = parent.winfo_screenwidth()
        screen_height = parent.winfo_screenheight()
        x = (screen_width - splash_width) // 2
        y = (screen_height - splash_height) // 2
        self.splash.geometry(f'{splash_width}x{splash_height}+{x}+{y}')

        # Ensure splash screen is on top
        self.splash.lift()
        self.splash.focus_force()

    def update_status(self, text):
        """Update the loading text"""
        self.loading_label.config(text=text)

    def destroy(self):
        """Destroy the splash screen"""
        self.splash.destroy() 