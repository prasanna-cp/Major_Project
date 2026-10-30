import cv2
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# -------- Dataset --------
class VideoDataset(Dataset):

    def __init__(self, folder):
        self.samples = []
        self.transform = transforms.ToTensor()

        videos = os.listdir(folder)

        for vid in videos:

            label = 1 if "accident" in vid.lower() else 0
            path = os.path.join(folder, vid)

            cap = cv2.VideoCapture(path)
            count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if count % 15 == 0:
                    frame = cv2.resize(frame,(224,224))
                    self.samples.append((frame,label))

                count += 1

            cap.release()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,idx):
        img,label = self.samples[idx]
        img = self.transform(img)
        return img, torch.tensor(label)

dataset = VideoDataset("Video")
loader = DataLoader(dataset,batch_size=16,shuffle=True)

# -------- Model --------
class AccidentCNN(nn.Module):
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
            nn.Linear(32*56*56,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )

    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.fc(x)

model = AccidentCNN()

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)

# -------- Training --------
for epoch in range(5):
    for imgs,labels in loader:
        opt.zero_grad()
        out = model(imgs)
        loss = loss_fn(out,labels)
        loss.backward()
        opt.step()

torch.save(model.state_dict(),"model/accident_model.pth")

print("✅ Accident model trained")