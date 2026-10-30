import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# IMPORTANT → Folder names automatically become labels
dataset = datasets.ImageFolder("Dataset1", transform=transform)

print(dataset.classes)   # ['High','Low']

loader = DataLoader(dataset, batch_size=16, shuffle=True)

class SeverityCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(32*56*56,128),
            nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

model = SeverityCNN()

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for img,label in loader:
        opt.zero_grad()
        out = model(img)
        loss = loss_fn(out,label)
        loss.backward()
        opt.step()

torch.save(model.state_dict(),"model/severity_model.pth")
print("✅ Model saved")