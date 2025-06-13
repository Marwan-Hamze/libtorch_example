# save_model.py
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
example_input = torch.randn(1, 3)

# Saving with Torch.jit.trace
traced_model = torch.jit.trace(model, example_input)
traced_model.save("simple_model.pt")

# Saving with Torch.save
torch.save(model, "torch_save_simple_model.pt")
