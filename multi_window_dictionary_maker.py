import tkinter as tk
from tkinter import simpledialog
import json
import pyautogui


class DraggableLabel(tk.Label):
    def __init__(self, parent, text, **kwargs):
        super().__init__(parent, text=text, **kwargs)
        self.bind('<Button-1>', self.start_drag)
        self.bind('<B1-Motion>', self.on_drag)
        self.bind('<ButtonRelease-1>', self.stop_drag)
        self.bind('<Shift-Button-1>', self.shift_select)
        self.bind('<Control-Button-1>', self.ctrl_select)

        self.selected = False
        self.parent = parent

    def toggle_selection(self):
        self.selected = not self.selected
        self.configure(bg='lightblue' if self.selected else 'white')

    def shift_select(self, event):
        if not hasattr(self.parent, 'last_selected'):
            self.parent.last_selected = self
            self.toggle_selection()
            return

        all_labels = [w for w in self.parent.winfo_children() if isinstance(w, DraggableLabel)]
        start_idx = all_labels.index(self.parent.last_selected)
        end_idx = all_labels.index(self)

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        for label in all_labels[start_idx:end_idx + 1]:
            if not label.selected:
                label.toggle_selection()

    def ctrl_select(self, event):
        self.toggle_selection()
        self.parent.last_selected = self

    def start_drag(self, event):
        if not self.selected:
            for widget in self.parent.winfo_children():
                if isinstance(widget, DraggableLabel) and widget.selected:
                    widget.toggle_selection()
            self.toggle_selection()

        self.parent.last_selected = self

        selected_items = [w for w in self.parent.winfo_children()
                          if isinstance(w, DraggableLabel) and w.selected]

        self.drag_window = tk.Toplevel()
        self.drag_window.overrideredirect(True)
        self.drag_window.attributes('-alpha', 0.7)

        for item in selected_items:
            tk.Label(self.drag_window, text=item.cget('text'),
                     bg='lightblue', padx=10, pady=5).pack()

        self.drag_window.geometry(f"+{event.x_root}+{event.y_root}")

    def on_drag(self, event):
        if hasattr(self, 'drag_window'):
            x = event.x_root
            y = event.y_root
            self.drag_window.geometry(f"+{x}+{y}")

    def stop_drag(self, event):
        if hasattr(self, 'drag_window'):
            selected_items = [w.cget('text') for w in self.parent.winfo_children()
                              if isinstance(w, DraggableLabel) and w.selected]

            x, y = event.x_root, event.y_root

            # Get only valid box windows
            valid_boxes = [box for box in BoxWindow.instances if box.winfo_exists()]

            for box in valid_boxes:
                try:
                    box_x = box.winfo_x()
                    box_y = box.winfo_y()
                    box_width = box.winfo_width()
                    box_height = box.winfo_height()

                    if (box_x <= x <= box_x + box_width and
                            box_y <= y <= box_y + box_height):
                        for item in selected_items:
                            box.add_item(item)
                        break
                except tk.TclError:
                    # If we can't get the window info, skip this box
                    continue

            self.drag_window.destroy()
            del self.drag_window

            for widget in self.parent.winfo_children():
                if isinstance(widget, DraggableLabel) and widget.selected:
                    widget.toggle_selection()


class BoxWindow(tk.Toplevel):
    instances = []

    def __init__(self, parent, label, x, y, items=None):
        super().__init__(parent)
        BoxWindow.instances.append(self)

        self.label = label
        self.title(f"Box: {label}")
        self.geometry(f"300x300+{x}+{y}")
        self.attributes('-topmost', True)

        tk.Label(self, text=label, font=('Arial', 12, 'bold')).pack(pady=5)

        self.items_frame = tk.Frame(self)
        self.items_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.items = items if items else []
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_items_display()

    def add_item(self, item):
        if item not in self.items:
            self.items.append(item)
            self.update_items_display()

    def update_items_display(self):
        for widget in self.items_frame.winfo_children():
            widget.destroy()

        for item in self.items:
            item_frame = tk.Frame(self.items_frame)
            item_frame.pack(fill=tk.X, pady=2)
            tk.Label(item_frame, text=item).pack(side=tk.LEFT)
            tk.Button(item_frame, text="×", command=lambda i=item: self.remove_item(i),
                      fg='red').pack(side=tk.RIGHT)

    def remove_item(self, item):
        if item in self.items:
            self.items.remove(item)
            self.update_items_display()

    def on_closing(self):
        BoxWindow.instances.remove(self)
        self.destroy()


class DictionaryBuilder:
    def __init__(self, root, input_dict=None, labels=None):
        self.root = root
        self.root.title("Dictionary Builder")
        self.result = None
        self.running = True

        # Clear any existing BoxWindow instances
        BoxWindow.instances.clear()

        # Use default labels if none provided
        self.labels = labels if labels is not None else ['item1', 'item2', 'item3', 'item4', 'item5', 'item6']

        control_panel = tk.Frame(root)
        control_panel.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(control_panel, text="New Box",
                  command=self.create_new_box).pack(side=tk.LEFT)
        tk.Button(control_panel, text="Get Dictionary",
                  command=self.show_dictionary).pack(side=tk.LEFT, padx=5)
        tk.Button(control_panel, text="Save and Close",
                  command=self.save_and_close).pack(side=tk.RIGHT)

        self.items_frame = tk.Frame(root)
        self.items_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.items_frame.last_selected = None

        if input_dict:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            x = 100
            y = 100
            for label, items in input_dict.items():
                BoxWindow(self.root, label, x, y, items)
                x += 350
                if x > screen_width - 300:
                    x = 100
                    y += 350
                if y > screen_height - 300:
                    break

        for item in self.labels:
            label = DraggableLabel(
                self.items_frame,
                text=item,
                relief=tk.RAISED,
                bg='white',
                padx=10,
                pady=5
            )
            label.pack(fill=tk.X, pady=2)

        self.root.bind('<Return>', self.create_new_box_with_enter)

        # Add protocol handler for window close button
        self.root.protocol("WM_DELETE_WINDOW", self.save_and_close)

    def create_new_box(self):
        label = simpledialog.askstring("New Box", "Enter box label:")
        if label:
            x, y = pyautogui.position()
            BoxWindow(self.root, label, x, y)

    def create_new_box_with_enter(self, event):
        self.create_new_box()

    def get_current_dictionary(self):
        result = {}
        # Only include boxes that still exist
        valid_boxes = [box for box in BoxWindow.instances if box.winfo_exists()]
        for box in valid_boxes:
            result[box.label] = box.items
        return result

    def show_dictionary(self):
        result = self.get_current_dictionary()

        result_window = tk.Toplevel(self.root)
        result_window.title("Dictionary Result")
        result_window.geometry("400x300")

        text_widget = tk.Text(result_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        formatted_dict = json.dumps(result, indent=2)
        text_widget.insert('1.0', formatted_dict)
        text_widget.config(state='disabled')

    def save_and_close(self):
        self.result = self.get_current_dictionary()
        self.running = False
        # Close all valid box windows
        valid_boxes = [box for box in BoxWindow.instances if box.winfo_exists()]
        for box in valid_boxes:
            box.destroy()
        self.root.destroy()

    def run(self):
        while self.running:
            try:
                self.root.update()
            except tk.TclError:  # Window has been destroyed
                break
        return self.result


def create_dictionary_builder(input_dict=None, labels=None):
    root = tk.Tk()
    app = DictionaryBuilder(root, input_dict, labels)
    root.geometry("300x400")
    return app.run()


if __name__ == "__main__":
    input_dict = {
        "Box 1": ["item1", "item2"],
        "Box 2": ["item3", "item4"],
        "Box 3": ["item5", "item6"],
    }

    # Example usage:
    result = create_dictionary_builder(input_dict)
    print("Resulting dictionary:", result)