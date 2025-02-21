import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import tkinter as tk
from tkinter import ttk, scrolledtext
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuginnGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Huginn-0125 Model Interface")
        
        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
        
        self.create_widgets()
        
    def create_widgets(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.root, text="Input", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=5)
        self.input_text.pack(fill="x")
        
        # Controls frame
        controls_frame = ttk.Frame(self.root, padding=10)
        controls_frame.pack(fill="x", padx=10)
        
        ttk.Label(controls_frame, text="Steps:").pack(side="left")
        self.steps_var = tk.StringVar(value="32")
        steps_entry = ttk.Entry(controls_frame, textvariable=self.steps_var, width=5)
        steps_entry.pack(side="left", padx=5)
        
        self.generate_btn = ttk.Button(controls_frame, text="Generate", command=self.generate_text)
        self.generate_btn.pack(side="left", padx=5)
        
        # Output frame
        output_frame = ttk.LabelFrame(self.root, text="Output", padding=10)
        output_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.output_text = scrolledtext.ScrolledText(output_frame)
        self.output_text.pack(fill="both", expand=True)
        
    def generate_text(self):
        try:
            input_text = self.input_text.get("1.0", "end-1c")
            steps = int(self.steps_var.get())
            
            # Disable the generate button while processing
            self.generate_btn['state'] = 'disabled'
            self.output_text.delete("1.0", "end")
            self.root.update()
            
            # Encode input text and create attention mask
            encoded = self.tokenizer(input_text, 
                return_tensors="pt",
                add_special_tokens=True,
                return_attention_mask=True)
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
                
            # Initialize empty generated sequence with input
            generated = input_ids
            current_mask = attention_mask
            
            with torch.no_grad():
                self.model.eval()
                # Generate one token at a time
                for _ in range(steps):
                    outputs = self.model.generate(
                        generated,
                        attention_mask=current_mask,
                        max_new_tokens=1,
                        num_return_sequences=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=False
                    )
                    
                    # Update generated sequence
                    generated = outputs.sequences
                    # Extend attention mask for the new token
                    current_mask = torch.cat([current_mask, torch.ones((1, 1), dtype=torch.long)], dim=1)
                    
                    # Decode and display the current state
                    current_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                    self.output_text.delete("1.0", "end")
                    self.output_text.insert("1.0", current_text)
                    self.root.update()
            
        except Exception as e:
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", f"Error: {str(e)}")
        finally:
            self.generate_btn['state'] = 'normal'

if __name__ == "__main__":
    root = tk.Tk()
    app = HuginnGUI(root)
    root.mainloop()
