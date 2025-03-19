import tkinter as tk
from tkinter import ttk


class GroupManager:
    def __init__(self, labels, group_capture, group_association, callback=None):
        self.labels = labels
        self.group_capture = group_capture.copy()
        self.group_association = group_association.copy()
        self.callback = callback

        # Create and hide root window
        self.root = tk.Tk()
        self.root.withdraw()

        # Create main window
        self.window = tk.Toplevel(self.root)
        self.window.title("Update Groups")
        self.window.geometry("800x500")

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.window, padding="10 10 10 10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create frames for each column
        self.labels_frame = ttk.Frame(self.main_frame)
        self.capture_frame = ttk.Frame(self.main_frame)
        self.association_frame = ttk.Frame(self.main_frame)

        # Grid layout
        for i, frame in enumerate([self.labels_frame, self.capture_frame, self.association_frame]):
            frame.grid(row=0, column=i, padx=10, pady=10, sticky="nsew")
            self.main_frame.columnconfigure(i, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # Create listboxes
        self.setup_listboxes()
        self.setup_buttons()

    def setup_listboxes(self):
        # Labels listbox
        ttk.Label(self.labels_frame, text="Available Labels").pack(pady=(0, 5))
        self.listbox_labels = self.create_listbox(self.labels_frame)

        # Group Inst listbox
        ttk.Label(self.capture_frame, text="Capture Group").pack(pady=(0, 5))
        self.listbox_capture = self.create_listbox(self.capture_frame)

        # Group Other listbox
        ttk.Label(self.association_frame, text="Association Group").pack(pady=(0, 5))
        self.listbox_association = self.create_listbox(self.association_frame)

        # Populate listboxes
        self.populate_listboxes()

    def create_listbox(self, parent):
        listbox = tk.Listbox(parent, selectmode=tk.MULTIPLE)
        listbox.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        listbox.drag_start = None
        listbox.selection_anchor = None
        listbox.bind('<Button-1>', lambda e, lb=listbox: self.on_click(e, lb))
        listbox.bind('<B1-Motion>', lambda e, lb=listbox: self.on_drag(e, lb))
        listbox.bind('<ButtonRelease-1>', lambda e, lb=listbox: self.on_release(e, lb))

        return listbox

    def populate_listboxes(self):
        # Clear existing items
        for listbox in [self.listbox_labels, self.listbox_capture, self.listbox_association]:
            listbox.delete(0, tk.END)

        # Populate available labels
        for item in self.labels:
            if item not in self.group_capture and item not in self.group_association:
                self.listbox_labels.insert(tk.END, item)

        # Populate group lists
        for item in self.group_capture:
            self.listbox_capture.insert(tk.END, item)
        for item in self.group_association:
            self.listbox_association.insert(tk.END, item)

    def setup_buttons(self):
        # Button frame
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=10)

        # Movement buttons
        ttk.Button(button_frame, text="To Available Labels",
            command=self.move_to_available).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="To Capture Group",
                  command=self.move_to_capture).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="To Association Group",
                  command=self.move_to_association).pack(side=tk.LEFT, padx=5)


        # Save and close button
        ttk.Button(self.main_frame, text="Save and Close",
                  command=self.save_and_close).grid(row=2, column=1, pady=10)

    def move_to_capture(self):
        # Move from Available Labels or Association Group to Capture Group
        self.move_selected_items(self.listbox_labels, self.listbox_capture)
        self.move_selected_items(self.listbox_association, self.listbox_capture)

    def move_to_association(self):
        # Move from Available Labels or Capture Group to Association Group
        self.move_selected_items(self.listbox_labels, self.listbox_association)
        self.move_selected_items(self.listbox_capture, self.listbox_association)

    def move_to_available(self):
        # Move from Capture Group or Association Group to Available Labels
        self.move_selected_items(self.listbox_capture, self.listbox_labels)
        self.move_selected_items(self.listbox_association, self.listbox_labels)

    def move_selected_items(self, from_listbox, to_listbox):
        selected_indices = from_listbox.curselection()
        selected_items = [from_listbox.get(i) for i in selected_indices]
        for item in selected_items:
            to_listbox.insert(tk.END, item)
        for index in reversed(selected_indices):
            from_listbox.delete(index)

    def on_click(self, event, listbox):
        listbox.drag_start = event.y
        index = listbox.nearest(event.y)
        if 0 <= index < listbox.size():
            listbox.selection_anchor = index
            if not (event.state & 0x0004):  # Check for Ctrl key
                listbox.selection_clear(0, tk.END)
            listbox.selection_set(index)

    def on_drag(self, event, listbox):
        if listbox.drag_start is None:
            return

        cur_index = listbox.nearest(event.y)
        if 0 <= cur_index < listbox.size():
            if not (event.state & 0x0004):  # Check for Ctrl key
                listbox.selection_clear(0, tk.END)
            if listbox.selection_anchor is not None:
                start = min(listbox.selection_anchor, cur_index)
                end = max(listbox.selection_anchor, cur_index)
                listbox.selection_set(start, end)

    def on_release(self, event, listbox):
        listbox.drag_start = None

    def save_and_close(self):
        self.group_capture = list(self.listbox_capture.get(0, tk.END))
        self.group_association = list(self.listbox_association.get(0, tk.END))
        if self.callback:
            self.callback(self.group_capture, self.group_association)
        self.window.destroy()
        self.root.destroy()

    def run(self):
        self.window.mainloop()


def main():
    # Sample data
    all_labels = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5",
                  "Item 6", "Item 7", "Item 8", "Item 9", "Item 10"]
    group_capture = ["Item 2", "Item 4"]
    group_association = ["Item 7", "Item 9"]

    # Callback function
    def on_groups_updated(new_group_capture, new_group_association):
        print("\nUpdated Groups:")
        print("Capture Group:", new_group_capture)
        print("Association Group:", new_group_association)

    # Create and run the group manager
    manager = GroupManager(all_labels, group_capture, group_association, on_groups_updated)
    manager.run()


if __name__ == "__main__":
    main()