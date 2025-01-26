import deepspeed
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

# Assuming you have your Transformer model implemented as `Transformer`
model = Transformer(
    n_vocab_len=37000,
    i_vocab_padding=0,
    d_model=512,
    device="cuda"
).to("cuda")

# Custom Dataset (Placeholder)
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Dummy data (replace with real data)
train_data = [
    (torch.randint(0, 37000, (50,)), torch.randint(0, 37000, (50,))) for _ in range(1000)
]
train_dataset = CustomDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Loss function
criterion = CrossEntropyLoss(ignore_index=0)

# DeepSpeed Configuration
ds_config = "ds_config.json"  # Path to your DeepSpeed config file
parameters = filter(lambda p: p.requires_grad, model.parameters())

# Initialize DeepSpeed
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=parameters,
    config=ds_config
)

# Training Loop
epochs = 3
for epoch in range(epochs):
    model_engine.train()
    for step, (src, tgt) in enumerate(train_dataloader):
        src, tgt = src.to("cuda"), tgt.to("cuda")
        output = model_engine(src, tgt)

        # Flatten the logits and labels for CrossEntropy
        logits = output.view(-1, output.size(-1))
        labels = tgt.view(-1)

        # Compute loss
        loss = criterion(logits, labels)

        # Backpropagation and optimization
        model_engine.backward(loss)
        model_engine.step()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")
