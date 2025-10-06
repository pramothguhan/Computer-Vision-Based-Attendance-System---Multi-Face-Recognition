import torch, torch.nn as nn, torch.nn.functional as F

class CelebrityIdentificationCNN(nn.Module):
    """Simple CNN with GAP head — efficient for 224x224."""
    def __init__(self, num_celebrities: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,3,padding=1);  self.bn1=nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,128,3,padding=1);self.bn2=nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,3,padding=1);self.bn3=nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,512,3,padding=1);self.bn4=nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2,2)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.5)
        self.fc1  = nn.Linear(512,512)
        self.cls  = nn.Linear(512, num_celebrities)

    def backbone(self,x):
        x=self.pool(F.relu(self.bn1(self.conv1(x))))
        x=self.pool(F.relu(self.bn2(self.conv2(x))))
        x=self.pool(F.relu(self.bn3(self.conv3(x))))
        x=self.pool(F.relu(self.bn4(self.conv4(x))))
        return x
    def extract_features(self,x):
        x=self.backbone(x)
        x=self.gap(x).flatten(1)
        x=F.relu(self.fc1(x)); x=self.drop(x)
        return x
    def forward(self,x):
        x=self.extract_features(x)
        return self.cls(x)

def create_model(n): return CelebrityIdentificationCNN(n)
def count_parameters(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)