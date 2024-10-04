import tkinter as tk
from tkinter import simpledialog


class DragDropDictBuilder:
    def __init__(self, root):
        self.root = root
        self.root.title("Dictionary Builder")

        # Main container
        self.container = tk.Frame(root)
        self.container.pack(fill=tk.BOTH, expand=True)

        # Canvas for boxes
        self.canvas = tk.Canvas(self.container, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Sidebar for list items
        self.sidebar = tk.Frame(self.container, bg='lightgray', width=150)
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        # List items (dummy data)
        self.list_items = ['list_item1', 'list_item2', 'list_item3', 'list_item4', 'list_item5']
        self.create_list_items()

        # Store boxes and their contents
        self.boxes = {}  # {box_id: {'label': label, 'items': [], 'coords': (x1,y1,x2,y2)}}

        # Drag and drop variables
        self.drawing_box = False
        self.start_x = None
        self.start_y = None
        self.dragged_item = None
        self.drag_text_id = None

        # Bind events
        self.canvas.bind('<Button-1>', self.start_box)
        self.canvas.bind('<B1-Motion>', self.draw_box)
        self.canvas.bind('<ButtonRelease-1>', self.end_box)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_list_items(self):
        """Create draggable labels for list items"""
        for item in self.list_items:
            label = tk.Label(self.sidebar, text=item, bg='white',
                             relief='raised', pady=5)
            label.pack(pady=2, padx=5, fill=tk.X)
            label.bind('<Button-1>', self.start_drag)
            label.bind('<B1-Motion>', self.drag)
            label.bind('<ButtonRelease-1>', self.drop)

    def start_box(self, event):
        """Start drawing a box"""
        if not self.dragged_item:  # Only start drawing if not dragging
            self.drawing_box = True
            self.start_x = event.x
            self.start_y = event.y
            self.temp_box = None

    def draw_box(self, event):
        """Draw box as mouse moves"""
        if self.drawing_box:
            if self.temp_box:
                self.canvas.delete(self.temp_box)
            self.temp_box = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline='black'
            )

    def end_box(self, event):
        """Finish drawing box and prompt for label"""
        if self.drawing_box:
            self.drawing_box = False
            if self.temp_box:
                # Get box dimensions
                x1, y1 = self.start_x, self.start_y
                x2, y2 = event.x, event.y

                # Ensure positive dimensions
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # Only create box if it has meaningful size
                if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:
                    label = simpledialog.askstring("Input", "Enter box label:")
                    if label:
                        # Create permanent box
                        box_id = self.canvas.create_rectangle(x1, y1, x2, y2,
                                                              outline='black', fill='white')
                        # Add label
                        self.canvas.create_text(x1 + 5, y1 + 5, text=label,
                                                anchor='nw', tags=f"label_{box_id}")
                        # Store box info
                        self.boxes[box_id] = {
                            'label': label,
                            'items': [],
                            'coords': (x1, y1, x2, y2)
                        }

                self.canvas.delete(self.temp_box)

    def start_drag(self, event):
        """Start dragging a list item"""
        widget = event.widget
        self.dragged_item = widget.cget('text')
        # Create a visual representation of the dragged item
        if self.drag_text_id:
            self.canvas.delete(self.drag_text_id)
        self.drag_text_id = self.canvas.create_text(
            event.x_root - self.canvas.winfo_rootx(),
            event.y_root - self.canvas.winfo_rooty(),
            text=self.dragged_item,
            fill='blue'
        )

    def drag(self, event):
        """Handle item dragging"""
        if self.dragged_item and self.drag_text_id:
            # Update the position of the dragged text
            canvas_x = event.x_root - self.canvas.winfo_rootx()
            canvas_y = event.y_root - self.canvas.winfo_rooty()
            self.canvas.coords(self.drag_text_id, canvas_x, canvas_y)

    def drop(self, event):
        """Handle dropping item into a box"""
        if self.dragged_item:
            # Get canvas coordinates
            canvas_x = event.x_root - self.canvas.winfo_rootx()
            canvas_y = event.y_root - self.canvas.winfo_rooty()

            # Check which box was dropped on
            for box_id, box_info in self.boxes.items():
                x1, y1, x2, y2 = box_info['coords']
                if x1 <= canvas_x <= x2 and y1 <= canvas_y <= y2:
                    # Add item to box if not already present
                    if self.dragged_item not in box_info['items']:
                        box_info['items'].append(self.dragged_item)
                        # Update the visual representation
                        self.update_box_items(box_id)
                    break

            # Clean up
            if self.drag_text_id:
                self.canvas.delete(self.drag_text_id)
                self.drag_text_id = None
            self.dragged_item = None

    def update_box_items(self, box_id):
        """Update the display of items in a box"""
        box_info = self.boxes[box_id]
        x1, y1, x2, y2 = box_info['coords']

        # Clear existing items (except the label)
        for item in self.canvas.find_all():
            if self.canvas.type(item) == 'text':
                if f"label_{box_id}" not in self.canvas.gettags(item):
                    ix1, iy1, ix2, iy2 = self.canvas.bbox(item)
                    if x1 <= ix1 <= x2 and y1 <= iy1 <= y2:
                        self.canvas.delete(item)

        # Display items
        y_offset = 25
        for item in box_info['items']:
            self.canvas.create_text(x1 + 5, y1 + y_offset,
                                    text=item, anchor='nw')
            y_offset += 20

    def get_dictionary(self):
        """Convert boxes to dictionary format"""
        return {box_info['label']: box_info['items']
                for box_info in self.boxes.values()}

    def on_closing(self):
        """Handle window closing"""
        print("Final Dictionary:", self.get_dictionary())
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = DragDropDictBuilder(root)
    root.geometry("800x600")
    root.mainloop()