import tkinter as tk
from tkinter import messagebox, filedialog

class TrainingDataGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NN Training Data Creator")

        self.input_count = tk.IntVar(value=2)
        self.output_count = tk.IntVar(value=1)
        self.data = []

        self.setup_controls()

    def reset(self):
        self.input_count.set(2)
        self.output_count.set(1)
        self.data.clear()
        self.data_listbox.delete(0, tk.END)

    def setup_controls(self):
        config_frame = tk.Frame(self.root)
        config_frame.pack(pady=10)

        tk.Label(config_frame, text="Inputs:").grid(row=0, column=0)
        tk.Entry(config_frame, textvariable=self.input_count, width=3).grid(row=0, column=1)

        tk.Label(config_frame, text="Outputs:").grid(row=0, column=2)
        tk.Entry(config_frame, textvariable=self.output_count, width=3).grid(row=0, column=3)

        tk.Button(config_frame, text="Set", command=self.create_io_fields).grid(row=0, column=4, padx=10)

        tk.Button(config_frame, text="Reset", command=self.reset).grid(row=0, column=5)

        self.io_frame = tk.Frame(self.root)
        self.io_frame.pack(pady=5)

        self.data_listbox = tk.Listbox(self.root, width=40)
        self.data_listbox.pack(pady=10)

        button_frame = tk.Frame(self.root)
        button_frame.pack()

        tk.Button(button_frame, text="Add Pair", command=self.add_pair).grid(row=0, column=0, padx=5)
        tk.Button(button_frame, text="Save", command=self.save_data).grid(row=0, column=1, padx=5)

    def create_io_fields(self):
        for widget in self.io_frame.winfo_children():
            widget.destroy()

        self.input_vars = [tk.IntVar(value=0) for _ in range(self.input_count.get())]
        self.output_vars = [tk.IntVar(value=0) for _ in range(self.output_count.get())]

        tk.Label(self.io_frame, text="Inputs:").grid(row=0, column=0)
        for i, var in enumerate(self.input_vars):
            b = tk.Checkbutton(self.io_frame, variable=var)
            b.grid(row=0, column=i + 1)

        tk.Label(self.io_frame, text="Outputs:").grid(row=1, column=0)
        for i, var in enumerate(self.output_vars):
            b = tk.Checkbutton(self.io_frame, variable=var)
            b.grid(row=1, column=i + 1)

    def add_pair(self):
        input_vals = [v.get() for v in self.input_vars]
        output_vals = [v.get() for v in self.output_vars]
        pair = f"{input_vals} {output_vals}"
        self.data.append(pair)
        self.data_listbox.insert(tk.END, pair)

    def save_data(self):
        if not self.data:
            messagebox.showwarning("No Data", "You haven't added any pairs.")
            return
        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if filepath:
            with open(filepath, "w") as f:
                f.writelines(line + "\n" for line in self.data)
            messagebox.showinfo("Saved", f"Saved {len(self.data)} pairs to {filepath}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrainingDataGUI(root)
    root.mainloop()
