import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import re
import os

class MethodOrganizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Code Organizer")
        self.root.geometry("1200x800")
        
        self.source_file = None
        self.regions_file = None
        self.methods = []
        self.regions = {}
        self.method_to_region = {}
        
        self.create_ui()
    
    def create_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(file_frame, text="Select Source File", command=self.select_source_file).grid(row=0, column=0, padx=5, pady=5)
        self.source_label = ttk.Label(file_frame, text="No file selected")
        self.source_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(file_frame, text="Select Regions File", command=self.select_regions_file).grid(row=1, column=0, padx=5, pady=5)
        self.regions_label = ttk.Label(file_frame, text="No file selected")
        self.regions_label.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Button(file_frame, text="Load Files", command=self.load_files).grid(row=2, column=0, columnspan=2, pady=10)
        
        # Organizer frame
        self.organizer_frame = ttk.LabelFrame(main_frame, text="Method Organizer", padding="10")
        self.organizer_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Split into two panes
        self.paned_window = ttk.PanedWindow(self.organizer_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Left pane - unassigned methods
        left_frame = ttk.LabelFrame(self.paned_window, text="Unassigned Methods")
        self.paned_window.add(left_frame, weight=1)
        
        self.unassigned_listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED)
        self.unassigned_listbox.pack(fill=tk.BOTH, expand=True)
        self.unassigned_listbox.bind("<B1-Motion>", self.on_drag)
        self.unassigned_listbox.bind("<ButtonRelease-1>", self.on_drop)
        
        # Right pane - regions
        right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(right_frame, weight=2)
        
        # Canvas for scrolling
        self.canvas = tk.Canvas(right_frame)
        scrollbar = ttk.Scrollbar(right_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Frame inside canvas for regions
        self.regions_container = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.regions_container, anchor=tk.NW)
        
        # Configure canvas scrolling
        self.regions_container.bind("<Configure>", self.on_frame_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Bottom frame for actions
        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="Save Organization", command=self.save_organization).pack(side=tk.RIGHT, padx=5)
        ttk.Button(action_frame, text="Apply to Source", command=self.apply_to_source).pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Drag and drop variables
        self.drag_data = {"widget": None, "item": None}
    
    def select_source_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Source File",
            filetypes=[("Python Files", "*.py"), ("All Files", "*.*")]
        )
        if file_path:
            self.source_file = file_path
            self.source_label.config(text=os.path.basename(file_path))
    
    def select_regions_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Regions File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if file_path:
            self.regions_file = file_path
            self.regions_label.config(text=os.path.basename(file_path))
    
    def load_files(self):
        if not self.source_file or not self.regions_file:
            messagebox.showerror("Error", "Please select both source and regions files")
            return
        
        try:
            # Extract methods from source file
            self.extract_methods_from_source()
            
            # Parse regions file
            self.parse_regions_file()
            
            # Check if all methods in regions file exist in source
            missing_methods = []
            for region, methods in self.regions.items():
                for method in methods:
                    if method not in self.methods and method.strip():
                        missing_methods.append(method)
            
            if missing_methods:
                messagebox.showwarning(
                    "Missing Methods", 
                    f"The following methods in the regions file were not found in the source:\n\n{', '.join(missing_methods)}"
                )
            
            # Create the organizer UI
            self.create_organizer_ui()
            
            self.status_var.set("Files loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load files: {str(e)}")
    
    def extract_methods_from_source(self):
        with open(self.source_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find class definition
        class_match = re.search(r'class\s+(\w+):', content)
        if not class_match:
            raise ValueError("No class definition found in source file")
        
        class_name = class_match.group(1)
        
        # Extract method definitions
        method_pattern = r'def\s+(\w+)\s*\('
        self.methods = re.findall(method_pattern, content)
        
        if not self.methods:
            raise ValueError("No methods found in source file")
    
    def parse_regions_file(self):
        with open(self.regions_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by region
        region_blocks = re.split(r'#\s*endregion', content)
        
        for block in region_blocks:
            if not block.strip():
                continue
            
            # Extract region name
            region_match = re.search(r'#\s*region\s+(.*?)$', block, re.MULTILINE)
            if not region_match:
                continue
            
            region_name = region_match.group(1).strip()
            
            # Extract methods
            lines = block.split('\n')
            methods = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and line != region_name:
                    methods.append(line)
            
            self.regions[region_name] = methods
            
            # Update method to region mapping
            for method in methods:
                if method:
                    self.method_to_region[method] = region_name
    
    def create_organizer_ui(self):
        # Clear existing UI
        for widget in self.regions_container.winfo_children():
            widget.destroy()
        
        self.unassigned_listbox.delete(0, tk.END)
        
        # Add unassigned methods
        unassigned_methods = [m for m in self.methods if m not in self.method_to_region]
        for method in unassigned_methods:
            self.unassigned_listbox.insert(tk.END, method)
        
        # Create region frames
        self.region_listboxes = {}
        
        for i, (region_name, methods) in enumerate(self.regions.items()):
            region_frame = ttk.LabelFrame(self.regions_container, text=region_name)
            region_frame.grid(row=i, column=0, sticky="ew", padx=5, pady=5)
            self.regions_container.columnconfigure(0, weight=1)
            
            # Add listbox for methods
            listbox = tk.Listbox(region_frame, selectmode=tk.EXTENDED, height=10)
            listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Add methods to listbox
            for method in methods:
                if method in self.methods:  # Only add methods that exist in source
                    listbox.insert(tk.END, method)
            
            # Bind drag and drop events
            listbox.bind("<B1-Motion>", self.on_drag)
            listbox.bind("<ButtonRelease-1>", self.on_drop)
            
            self.region_listboxes[region_name] = listbox
        
        # Update canvas scrollregion
        self.regions_container.update_idletasks()
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
    
    def on_frame_configure(self, event):
        # Update the scrollregion to encompass the inner frame
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def on_canvas_configure(self, event):
        # Update the width of the window to fill the canvas
        self.canvas.itemconfig(self.canvas_window, width=event.width)
    
    def on_drag(self, event):
        widget = event.widget
        if not self.drag_data["widget"]:
            # Get the selected items
            if not widget.curselection():
                return
            
            self.drag_data["widget"] = widget
            self.drag_data["items"] = widget.curselection()
            self.drag_data["values"] = [widget.get(i) for i in widget.curselection()]
    
    def on_drop(self, event):
        if not self.drag_data["widget"]:
            return
        
        # Find the widget under the cursor
        x, y = event.x_root, event.y_root
        target_widget = self.root.winfo_containing(x, y)
        
        # Check if target is a listbox
        if isinstance(target_widget, tk.Listbox) and target_widget != self.drag_data["widget"]:
            # Remove from source
            source_widget = self.drag_data["widget"]
            items = self.drag_data["items"]
            values = self.drag_data["values"]
            
            # Add to target
            for value in values:
                target_widget.insert(tk.END, value)
            
            # Remove from source (in reverse order to maintain indices)
            for i in sorted(items, reverse=True):
                source_widget.delete(i)
            
            # Update method to region mapping
            for value in values:
                # Find which region the target listbox belongs to
                for region_name, listbox in self.region_listboxes.items():
                    if listbox == target_widget:
                        self.method_to_region[value] = region_name
                        break
                
                # If target is unassigned listbox, remove from mapping
                if target_widget == self.unassigned_listbox:
                    if value in self.method_to_region:
                        del self.method_to_region[value]
        
        # Reset drag data
        self.drag_data = {"widget": None, "items": None, "values": None}
    
    def save_organization(self):
        # Create output text
        output = []
        
        for region_name, methods in self.regions.items():
            # Get methods from listbox
            listbox = self.region_listboxes[region_name]
            methods = [listbox.get(i) for i in range(listbox.size())]
            
            # Add region header
            output.append(f"# region {region_name}")
            
            # Add methods
            for method in methods:
                output.append(method)
            
            # Add region footer
            output.append("# endregion\n")
        
        # Save to file
        save_path = filedialog.asksaveasfilename(
            title="Save Organization",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(output))
            
            self.status_var.set(f"Organization saved to {os.path.basename(save_path)}")
    
    def apply_to_source(self):
        if not messagebox.askyesno("Confirm", "This will modify your source file. Continue?"):
            return
        
        try:
            with open(self.source_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create a backup
            backup_path = f"{self.source_file}.bak"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Extract method definitions with their content
            method_pattern = r'(def\s+(\w+)\s*\([^)]*\):(?:\s*"""[\s\S]*?""")?[\s\S]*?)(?=\n\s*def\s+\w+\s*\(|\Z)'
            method_matches = re.findall(method_pattern, content)
            
            method_contents = {name: code for code, name in method_matches}
            
            # Build new content
            new_content = []
            
            # Find class definition and everything before it
            class_match = re.search(r'(.*?class\s+\w+:.*?)(?=\s*def|\Z)', content, re.DOTALL)
            if class_match:
                new_content.append(class_match.group(1))
            
            # Add methods organized by region
            for region_name, listbox in self.region_listboxes.items():
                methods = [listbox.get(i) for i in range(listbox.size())]
                
                if methods:
                    new_content.append(f"\n    # region {region_name}")
                    
                    for method_name in methods:
                        if method_name in method_contents:
                            # Indent method content properly
                            method_code = method_contents[method_name]
                            new_content.append(method_code)
                    
                    new_content.append("    # endregion\n")
            
            # Add anything after the class
            class_end_match = re.search(r'(class\s+\w+:.*?)(\n\S.*)', content, re.DOTALL)
            if class_end_match and class_end_match.group(2):
                new_content.append(class_end_match.group(2))
            
            # Write to file
            with open(self.source_file, 'w', encoding='utf-8') as f:
                f.write("".join(new_content))
            
            self.status_var.set(f"Source file updated. Backup saved to {os.path.basename(backup_path)}")
            messagebox.showinfo("Success", f"Source file updated. Backup saved to {backup_path}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update source file: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MethodOrganizer(root)
    root.mainloop() 