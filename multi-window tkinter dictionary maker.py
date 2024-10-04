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

        # Find all labels between last selected and current
        all_labels = [w for w in self.parent.winfo_children() if isinstance(w, DraggableLabel)]
        start_idx = all_labels.index(self.parent.last_selected)
        end_idx = all_labels.index(self)

        # Swap if needed to ensure correct order
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # Select all labels in range
        for label in all_labels[start_idx:end_idx + 1]:
            if not label.selected:
                label.toggle_selection()

    def ctrl_select(self, event):
        self.toggle_selection()
        self.parent.last_selected = self

    def start_drag(self, event):
        if not self.selected:
            # If clicking an unselected item, clear other selections
            for widget in self.parent.winfo_children():
                if isinstance(widget, DraggableLabel) and widget.selected:
                    widget.toggle_selection()
            self.toggle_selection()

        self.parent.last_selected = self

        # Get all selected items
        selected_items = [w for w in self.parent.winfo_children()
                          if isinstance(w, DraggableLabel) and w.selected]

        # Create drag representation
        self.drag_window = tk.Toplevel()
        self.drag_window.overrideredirect(True)
        self.drag_window.attributes('-alpha', 0.7)

        # Show all selected items in drag window
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
            # Get all selected items
            selected_items = [w.cget('text') for w in self.parent.winfo_children()
                              if isinstance(w, DraggableLabel) and w.selected]

            x, y = event.x_root, event.y_root

            # Find if we dropped on any box window
            for box in BoxWindow.instances:
                box_x = box.winfo_x()
                box_y = box.winfo_y()
                box_width = box.winfo_width()
                box_height = box.winfo_height()

                if (box_x <= x <= box_x + box_width and
                        box_y <= y <= box_y + box_height):
                    # Add all selected items to the box
                    for item in selected_items:
                        box.add_item(item)
                    break

            self.drag_window.destroy()
            del self.drag_window

            # Clear selections after drop
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


class MainWindow:
    def __init__(self, root, input_dict=None):
        self.root = root
        self.root.title("Dictionary Builder")

        control_panel = tk.Frame(root)
        control_panel.pack(fill=tk.X, padx=10, pady=5)

        tk.Button(control_panel, text="New Box",
                  command=self.create_new_box).pack(side=tk.LEFT)
        tk.Button(control_panel, text="Get Dictionary",
                  command=self.show_dictionary).pack(side=tk.LEFT, padx=5)

        self.items_frame = tk.Frame(root)
        self.items_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Track last selected item for shift-select
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

        items = ['item1', 'item2', 'item3', 'item4', 'item5', 'item6']

        for item in items:
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

    def create_new_box(self):
        label = simpledialog.askstring("New Box", "Enter box label:")
        if label:
            x, y = pyautogui.position()
            BoxWindow(self.root, label, x, y)

    def create_new_box_with_enter(self, event):
        self.create_new_box()

    def show_dictionary(self):
        result = {}
        for box in BoxWindow.instances:
            result[box.label] = box.items

        result_window = tk.Toplevel(self.root)
        result_window.title("Dictionary Result")
        result_window.geometry("400x300")

        text_widget = tk.Text(result_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        formatted_dict = json.dumps(result, indent=2)
        text_widget.insert('1.0', formatted_dict)
        text_widget.config(state='disabled')


if __name__ == "__main__":
    input_dict = {
        "Box 1": ["item1", "item2"],
        "Box 2": ["item3", "item4"],
        "Box 3": ["item5", "item6"],
    }

    root = tk.Tk()
    app = MainWindow(root, input_dict)
    root.geometry("300x400")
    root.mainloop()