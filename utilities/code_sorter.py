import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import os
from typing import Dict, List, Tuple, Set
import re
import ast

class CodeSorterGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Code Sorter")
        self.geometry("800x600")

        # Create main container
        self.main_container = ttk.Frame(self)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # File selection frame
        self.create_file_selection_frame()

        # Region definitions frame
        self.create_region_definitions_frame()

        # Options frame
        self.create_options_frame()

        # Status frame
        self.create_status_frame()

        # Initialize CodeSorter instance
        self.code_sorter = None

    def create_file_selection_frame(self):
        file_frame = ttk.LabelFrame(self.main_container, text="File Selection", padding=5)
        file_frame.pack(fill=tk.X, pady=5)

        # Input file selection
        ttk.Label(file_frame, text="Input Python File:").grid(row=0, column=0, sticky=tk.W)
        self.input_file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.input_file_var).grid(row=0, column=1, sticky=tk.EW)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_file("input")).grid(row=0, column=2)

        # Output file selection
        ttk.Label(file_frame, text="Output Python File:").grid(row=1, column=0, sticky=tk.W)
        self.output_file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_file_var).grid(row=1, column=1, sticky=tk.EW)
        ttk.Button(file_frame, text="Browse", command=lambda: self.browse_file("output")).grid(row=1, column=2)

        file_frame.columnconfigure(1, weight=1)

    def create_region_definitions_frame(self):
        region_frame = ttk.LabelFrame(self.main_container, text="Region Definitions", padding=5)
        region_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Add text area for region definitions
        self.region_text = scrolledtext.ScrolledText(region_frame, wrap=tk.WORD, height=15)
        self.region_text.pack(fill=tk.BOTH, expand=True)

        # Add example placeholder text
        example_text = """Can you organize into regions using this format: 
        
# functions
# region Utilities
def utility_function1
def utility_function2

# class MyClass
# region Core
def method1
def method2

Just use the function/method names; exclude the body.
"""
        self.region_text.insert('1.0', example_text)

    def create_options_frame(self):
        options_frame = ttk.LabelFrame(self.main_container, text="Options", padding=5)
        options_frame.pack(fill=tk.X, pady=5)

        # Strict mode checkbox
        self.strict_mode = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Strict Mode", variable=self.strict_mode).pack(side=tk.LEFT)

        # Sort button
        ttk.Button(options_frame, text="Sort Code", command=self.sort_code).pack(side=tk.RIGHT)

    def create_status_frame(self):
        status_frame = ttk.LabelFrame(self.main_container, text="Status", padding=5)
        status_frame.pack(fill=tk.X, pady=5)

        self.status_text = scrolledtext.ScrolledText(status_frame, wrap=tk.WORD, height=6)
        self.status_text.pack(fill=tk.BOTH, expand=True)

    def browse_file(self, file_type):
        if file_type == "input":
            filename = filedialog.askopenfilename(filetypes=[("Python files", "*.py")])
            if filename:
                self.input_file_var.set(filename)
                # Automatically set output filename
                base_dir = os.path.dirname(filename)
                base_name = os.path.splitext(os.path.basename(filename))[0]
                self.output_file_var.set(os.path.join(base_dir, f"{base_name}_sorted.py"))
        else:
            filename = filedialog.asksaveasfilename(
                defaultextension=".py",
                filetypes=[("Python files", "*.py")]
            )
            if filename:
                self.output_file_var.set(filename)

    def update_status(self, message, error=False):
        self.status_text.insert(tk.END, f"\n{message}")
        if error:
            self.status_text.tag_add("error", "end-2c linestart", "end-1c")
            self.status_text.tag_config("error", foreground="red")
        self.status_text.see(tk.END)

    def sort_code(self):
        # Clear status
        self.status_text.delete('1.0', tk.END)

        # Validate inputs
        if not self.input_file_var.get() or not self.output_file_var.get():
            self.update_status("Error: Please select both input and output files.", error=True)
            return

        if not self.region_text.get('1.0', tk.END).strip():
            self.update_status("Error: Region definitions cannot be empty.", error=True)
            return

        try:
            # Initialize CodeSorter with region definitions
            self.code_sorter = CodeSorter(self.region_text.get('1.0', tk.END))

            # Sort the file
            self.code_sorter.sort_file(
                self.input_file_var.get(),
                self.output_file_var.get(),
                self.strict_mode.get()
            )

            self.update_status("Code successfully sorted!")

            # Ask if user wants to open the output file
            if messagebox.askyesno("Success", "Would you like to open the sorted file?"):
                os.startfile(self.output_file_var.get()) if os.name == 'nt' else \
                    os.system(f'xdg-open {self.output_file_var.get()}')

        except Exception as e:
            self.update_status(f"Error: {str(e)}", error=True)





class CodeSorter:
    def __init__(self, region_def_str: str):
        """
        Initialize the sorter with region definitions for functions and classes.

        Args:
            region_def_str (str): String containing region definitions
                Format:
                # functions
                # region RegionName
                def function1
                def function2
                # class ClassName
                # region RegionName
                def method1
                def method2
                ...
        """
        self.regions = self._parse_region_definitions(region_def_str)

    def _parse_region_definitions(self, region_def_str: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Parse the region definition string into a nested dictionary.

        Returns:
            Dict[str, Dict[str, List[str]]]: Dictionary mapping scope names to their region definitions
                'functions' key contains standalone functions
                class names contain their respective methods
        """
        regions = {'functions': {}}  # Add dedicated functions section
        current_scope = None
        current_region = None
        region_def_str.replace('\n# endregion','')
        for line in region_def_str.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('# functions'):
                current_scope = 'functions'
                regions['functions'] = {}
            elif line.startswith('# class'):
                current_scope = line.replace('# class', '').strip()
                regions[current_scope] = {}
            elif line.startswith('# region') and current_scope:
                current_region = line.replace('# region', '').strip()
                regions[current_scope][current_region] = []
            elif line.startswith('def ') and current_scope and current_region:
                func_name = line.split('(')[0].replace('def', '').strip()
                regions[current_scope][current_region].append(func_name)

        return regions

    def _extract_code_elements(self, source_code: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
        """
        Extract all standalone functions and classes with their methods from the source code.

        Returns:
            Tuple[Dict[str, str], Dict[str, Dict[str, str]]]:
                1. Dictionary mapping function names to their source code
                2. Dictionary mapping class names to their methods
        """
        functions = {}
        classes = {}
        source_lines = source_code.split('\n')

        class CodeVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                # Only capture top-level functions
                if isinstance(node.parent, ast.Module):
                    func_lines = source_lines[node.lineno - 1:node.end_lineno]
                    functions[node.name] = '\n'.join(func_lines)

            def visit_ClassDef(self, node):
                class_name = node.name
                classes[class_name] = {}

                # Get the full class source
                class_lines = source_lines[node.lineno - 1:node.end_lineno]
                classes[class_name]['__class_def__'] = '\n'.join(class_lines)

                # Visit all methods in the class
                for child in node.body:
                    if isinstance(child, ast.FunctionDef):
                        method_lines = source_lines[child.lineno - 1:child.end_lineno]
                        classes[class_name][child.name] = '\n'.join(method_lines)

        try:
            tree = ast.parse(source_code)
            # Add parent references to make it easier to identify top-level functions
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    child.parent = node
            visitor = CodeVisitor()
            visitor.visit(tree)
        except Exception as e:
            print(f"Error parsing source code: {e}")
            return {}, {}

        return functions, classes

    def _validate_code_elements(
        self,
        functions: Dict[str, str],
        classes: Dict[str, Dict[str, str]]
    ) -> Tuple[List[str], List[str]]:
        """
        Validate that all functions and methods match between region definitions and source code.

        Returns:
            Tuple[List[str], List[str]]: Lists of missing and undefined elements
        """
        missing_elements = []
        undefined_elements = []

        # Check standalone functions
        if 'functions' in self.regions:
            region_funcs: Set[str] = set()
            for funcs in self.regions['functions'].values():
                region_funcs.update(funcs)

            source_funcs = set(functions.keys())

            undefined = region_funcs - source_funcs
            missing = source_funcs - region_funcs

            undefined_elements.extend(f"function.{func}" for func in undefined)
            missing_elements.extend(f"function.{func}" for func in missing)

        # Check class methods
        for class_name, regions in self.regions.items():
            if class_name == 'functions':
                continue

            if class_name not in classes:
                print(f"Warning: Class '{class_name}' defined in regions but not found in source")
                continue

            region_methods: Set[str] = set()
            for methods in regions.values():
                region_methods.update(methods)

            source_methods = set(classes[class_name].keys()) - {'__class_def__'}

            undefined = region_methods - source_methods
            missing = source_methods - region_methods

            undefined_elements.extend(f"{class_name}.{method}" for method in undefined)
            missing_elements.extend(f"{class_name}.{method}" for method in missing)

        return missing_elements, undefined_elements

    def sort_file(self, input_file: str, output_file: str, strict: bool = True):
        """
        Sort standalone functions and class methods according to region definitions.

        Args:
            input_file (str): Path to input Python file
            output_file (str): Path to output Python file
            strict (bool): If True, raise error on mismatches. If False, warn and proceed
        """
        # Read the input file
        with open(input_file, 'r') as f:
            source_code = f.read()

        # Extract functions and classes
        functions, classes = self._extract_code_elements(source_code)

        # Validate code elements
        missing_elements, undefined_elements = self._validate_code_elements(functions, classes)

        if missing_elements or undefined_elements:
            error_msg = []
            if missing_elements:
                error_msg.append("Elements found in source but not in region definitions:")
                error_msg.extend(f"  - {element}" for element in missing_elements)
            if undefined_elements:
                error_msg.append("\nElements defined in regions but not found in source:")
                error_msg.extend(f"  - {element}" for element in undefined_elements)

            if strict:
                raise ValueError("\n".join(error_msg))
            else:
                print("Warning: Element mismatches found:\n" + "\n".join(error_msg))

        # Create the sorted output
        output_lines = []

        # Add any initial imports or code before the first function/class
        first_element_pattern = r'^.*(?:def|class)\s+\w+.*?:'
        first_element_match = re.search(first_element_pattern, source_code, re.MULTILINE)
        if first_element_match:
            header = source_code[:first_element_match.start()].rstrip()
            if header:
                output_lines.append(header + '\n')

        # Add standalone functions first
        if 'functions' in self.regions and self.regions['functions']:
            output_lines.append("\n# Standalone Functions")
            for region_name, func_names in self.regions['functions'].items():
                output_lines.append(f"\n# region {region_name}")

                for func_name in func_names:
                    if func_name in functions:
                        output_lines.append(f"\n{functions[func_name]}")

                output_lines.append("# endregion\n")

        # Process each class
        for class_name, regions in self.regions.items():
            if class_name == 'functions' or class_name not in classes:
                continue

            # Add the class definition
            class_def = classes[class_name]['__class_def__'].split('\n')[0]
            output_lines.append(f"\n{class_def}")

            # Add methods by region
            for region_name, method_names in regions.items():
                output_lines.append(f"\n    # region {region_name}")

                for method_name in method_names:
                    if method_name in classes[class_name]:
                        method_lines = classes[class_name][method_name].split('\n')
                        first_line = method_lines[0]
                        rest_lines = [f"    {line}" for line in method_lines[1:]]
                        indented_method = '\n'.join([first_line] + rest_lines)
                        output_lines.append(f"\n{indented_method}")

                output_lines.append("    # endregion\n")

        # If not in strict mode, append any unorganized elements
        if not strict and missing_elements:
            # Add unorganized functions
            output_lines.append("\n# region Unorganized Functions")
            region_funcs = set()
            if 'functions' in self.regions:
                for funcs in self.regions['functions'].values():
                    region_funcs.update(funcs)

            for func_name, func_code in functions.items():
                if func_name not in region_funcs:
                    output_lines.append(f"\n{func_code}")
            output_lines.append("# endregion\n")

            # Add unorganized methods
            for class_name, class_data in classes.items():
                region_methods = set()
                if class_name in self.regions:
                    for methods in self.regions[class_name].values():
                        region_methods.update(methods)

                output_lines.append(f"\n    # region Unorganized Methods")
                for method_name, method_code in class_data.items():
                    if method_name != '__class_def__' and method_name not in region_methods:
                        method_lines = method_code.split('\n')
                        first_line = method_lines[0]
                        rest_lines = [f"    {line}" for line in method_lines[1:]]
                        indented_method = '\n'.join([first_line] + rest_lines)
                        output_lines.append(f"\n{indented_method}")
                output_lines.append("    # endregion\n")

        # Write the sorted code to output file
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))


def main():
    app = CodeSorterGUI()
    app.mainloop()


if __name__ == "__main__":
    main()