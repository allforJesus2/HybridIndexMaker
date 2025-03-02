import json
import tkinter as tk
from tkinter import ttk, messagebox
import os

class TextCorrections:
    def __init__(self, settings_path=''):
        self.settings_path = settings_path
        self.corrections = {}
        self.load_corrections()

    def load_corrections(self):
        """Load correction rules from JSON file"""
        corrections_file = os.path.join(self.settings_path, 'text_corrections.json')
        try:
            if os.path.exists(corrections_file):
                with open(corrections_file, 'r') as f:
                    self.corrections = json.load(f)
        except Exception as e:
            print(f"Error loading corrections: {e}")
            self.corrections = {}

    def save_corrections(self):
        """Save correction rules to JSON file"""
        corrections_file = os.path.join(self.settings_path, 'text_corrections.json')
        try:
            with open(corrections_file, 'w') as f:
                json.dump(self.corrections, f, indent=2)
        except Exception as e:
            print(f"Error saving corrections: {e}")

    def apply_corrections(self, text):
        """Apply correction rules to text"""
        if not text:
            return text
            
        corrected = text
        for wrong, right in self.corrections.items():
            corrected = corrected.replace(wrong, right)
        return corrected

class TextCorrectionsEditor:
    def __init__(self, parent, text_corrections):
        self.window = tk.Toplevel(parent)
        self.window.title("Text Corrections Editor")
        self.text_corrections = text_corrections
        
        # Create main frame
        self.frame = ttk.Frame(self.window, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create table
        self.create_table()
        
        # Create input fields
        self.create_input_fields()
        
        # Create buttons
        self.create_buttons()
        
        # Center window
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f'+{x}+{y}')

    def create_table(self):
        # Create Treeview
        columns = ('wrong', 'right')
        self.tree = ttk.Treeview(self.frame, columns=columns, show='headings')
        
        # Define headings
        self.tree.heading('wrong', text='Wrong Text')
        self.tree.heading('right', text='Correct Text')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Grid layout
        self.tree.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=5, pady=5)
        scrollbar.grid(row=0, column=2, sticky='ns')
        
        # Populate table
        self.refresh_table()
        
        # Bind selection event
        self.tree.bind('<<TreeviewSelect>>', self.on_select)

    def create_input_fields(self):
        # Wrong text input
        ttk.Label(self.frame, text="Wrong Text:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.wrong_text = ttk.Entry(self.frame)
        self.wrong_text.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=2)
        
        # Right text input
        ttk.Label(self.frame, text="Correct Text:").grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        self.right_text = ttk.Entry(self.frame)
        self.right_text.grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)

    def create_buttons(self):
        button_frame = ttk.Frame(self.frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Add/Update", command=self.add_correction).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Delete", command=self.delete_correction).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save", command=self.save_changes).pack(side=tk.LEFT, padx=5)

    def refresh_table(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Add all corrections
        for wrong, right in self.text_corrections.corrections.items():
            self.tree.insert('', tk.END, values=(wrong, right))

    def on_select(self, event):
        # Get selected item
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            values = item['values']
            
            # Update entry fields
            self.wrong_text.delete(0, tk.END)
            self.right_text.delete(0, tk.END)
            self.wrong_text.insert(0, values[0])
            self.right_text.insert(0, values[1])

    def add_correction(self):
        wrong = self.wrong_text.get().strip()
        right = self.right_text.get().strip()
        
        if not wrong or not right:
            messagebox.showerror("Error", "Both fields must be filled")
            return
            
        self.text_corrections.corrections[wrong] = right
        self.refresh_table()
        
        # Clear input fields
        self.wrong_text.delete(0, tk.END)
        self.right_text.delete(0, tk.END)

    def delete_correction(self):
        selection = self.tree.selection()
        if not selection:
            messagebox.showerror("Error", "Please select an item to delete")
            return
            
        item = self.tree.item(selection[0])
        wrong_text = item['values'][0]
        
        if wrong_text in self.text_corrections.corrections:
            del self.text_corrections.corrections[wrong_text]
            self.refresh_table()

    def save_changes(self):
        self.text_corrections.save_corrections()
        messagebox.showinfo("Success", "Changes saved successfully") 