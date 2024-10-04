import tkinter as tk
from tkinter import messagebox


class GroupInputApp:
    def __init__(self, master, groups):
        self.master = master
        self.groups = groups
        self.result = {}

        master.title("Group Input App")

        self.entries = {}
        for i, (key, value) in enumerate(groups.items()):
            tk.Label(master, text=f"{key}:").grid(row=i, column=0, sticky="e", padx=5, pady=5)
            entry = tk.Entry(master)
            entry.grid(row=i, column=1, padx=5, pady=5)
            entry.insert(0, value)
            self.entries[key] = entry

        tk.Button(master, text="Submit", command=self.on_submit).grid(row=len(groups), column=0, columnspan=2, pady=10)

        #master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_submit(self):
        for group, entry in self.entries.items():
            value = entry.get()
            if value:  # Only add non-empty values
                self.result[group] = value

        messagebox.showinfo("Success", "Inputs submitted successfully!")
        print(self.result)

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.master.quit()
