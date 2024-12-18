import re
import ast
from typing import Dict, List, Tuple, Set


class MethodSorter:
    def __init__(self, region_def_str: str):
        """
        Initialize the sorter with region definitions for multiple classes.

        Args:
            region_def_str (str): String containing class-specific region definitions
                Format:
                # class ClassName
                # region RegionName
                def method1
                def method2
                # class AnotherClass
                ...
        """
        self.class_regions = self._parse_region_definitions(region_def_str)

    def _parse_region_definitions(self, region_def_str: str) -> Dict[str, Dict[str, List[str]]]:
        """
        Parse the region definition string into a nested dictionary.

        Returns:
            Dict[str, Dict[str, List[str]]]: Dictionary mapping class names to their region definitions
        """
        class_regions = {}
        current_class = None
        current_region = None

        for line in region_def_str.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('# class'):
                current_class = line.replace('# class', '').strip()
                class_regions[current_class] = {}
            elif line.startswith('# region') and current_class:
                current_region = line.replace('# region', '').strip()
                class_regions[current_class][current_region] = []
            elif line.startswith('def ') and current_class and current_region:
                method_name = line.split('(')[0].replace('def', '').strip()
                class_regions[current_class][current_region].append(method_name)

        return class_regions

    def _extract_classes_and_methods(self, source_code: str) -> Dict[str, Dict[str, str]]:
        """
        Extract all classes and their methods from the source code.

        Returns:
            Dict[str, Dict[str, str]]: Dictionary mapping class names to their methods
        """
        classes = {}
        source_lines = source_code.split('\n')

        class ClassMethodVisitor(ast.NodeVisitor):
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
            visitor = ClassMethodVisitor()
            visitor.visit(tree)
        except Exception as e:
            print(f"Error parsing source code: {e}")
            return {}

        return classes

    def _validate_methods(self, class_dict: Dict[str, Dict[str, str]]) -> Tuple[List[str], List[str]]:
        """
        Validate that all methods match between region definitions and source code.

        Returns:
            Tuple[List[str], List[str]]: Lists of missing and undefined methods
        """
        missing_methods = []  # Methods in source but not in regions
        undefined_methods = []  # Methods in regions but not in source

        for class_name, regions in self.class_regions.items():
            if class_name not in class_dict:
                print(f"Warning: Class '{class_name}' defined in regions but not found in source")
                continue

            # Get all methods from regions
            region_methods: Set[str] = set()
            for methods in regions.values():
                region_methods.update(methods)

            # Get all methods from source (excluding __class_def__)
            source_methods = set(class_dict[class_name].keys()) - {'__class_def__'}

            # Find mismatches
            undefined = region_methods - source_methods
            missing = source_methods - region_methods

            undefined_methods.extend(f"{class_name}.{method}" for method in undefined)
            missing_methods.extend(f"{class_name}.{method}" for method in missing)

        return missing_methods, undefined_methods

    def sort_file(self, input_file: str, output_file: str, strict: bool = True):
        """
        Sort methods in multiple classes according to region definitions.

        Args:
            input_file (str): Path to input Python file
            output_file (str): Path to output Python file
            strict (bool): If True, raise error on mismatches. If False, warn and proceed
        """
        # Read the input file
        with open(input_file, 'r') as f:
            source_code = f.read()

        # Extract all classes and their methods
        class_dict = self._extract_classes_and_methods(source_code)

        # Validate methods
        missing_methods, undefined_methods = self._validate_methods(class_dict)

        if missing_methods or undefined_methods:
            error_msg = []
            if missing_methods:
                error_msg.append("Methods found in source but not in region definitions:")
                error_msg.extend(f"  - {method}" for method in missing_methods)
            if undefined_methods:
                error_msg.append("\nMethods defined in regions but not found in source:")
                error_msg.extend(f"  - {method}" for method in undefined_methods)

            if strict:
                raise ValueError("\n".join(error_msg))
            else:
                print("Warning: Method mismatches found:\n" + "\n".join(error_msg))

        # Create the sorted output (same as original implementation)
        output_lines = []

        # Add any initial imports or code before the first class
        first_class_match = re.search(r'^.*class\s+\w+.*?:', source_code, re.MULTILINE)
        if first_class_match:
            header = source_code[:first_class_match.start()].rstrip()
            if header:
                output_lines.append(header + '\n')

        # Process each class
        for class_name, regions in self.class_regions.items():
            if class_name not in class_dict:
                continue

            # Add the class definition
            class_def = class_dict[class_name]['__class_def__'].split('\n')[0]
            output_lines.append(f"\n{class_def}")

            # Add methods by region
            for region_name, method_names in regions.items():
                output_lines.append(f"\n    # region {region_name}")

                for method_name in method_names:
                    if method_name in class_dict[class_name]:
                        method_lines = class_dict[class_name][method_name].split('\n')
                        first_line = method_lines[0]
                        rest_lines = [f"    {line}" for line in method_lines[1:]]
                        indented_method = '\n'.join([first_line] + rest_lines)
                        output_lines.append(f"\n{indented_method}")

                output_lines.append("    # endregion\n")

        # If not in strict mode, append any methods that weren't in region definitions
        if not strict and missing_methods:
            for class_name, class_data in class_dict.items():
                region_methods = set()
                if class_name in self.class_regions:
                    for methods in self.class_regions[class_name].values():
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
    # Sort Python methods into regions based on a definition file
    region_file = 'regions.txt'
    input_file = 'test.py'
    output_file = 'output.py'

    # Read region definitions
    with open(region_file, 'r') as f:
        region_def_str = f.read()

    # Create sorter and process file
    sorter = MethodSorter(region_def_str)
    sorter.sort_file(input_file, output_file)


if __name__ == "__main__":
    main()