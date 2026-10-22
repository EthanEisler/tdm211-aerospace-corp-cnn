import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes, in_ch=1, widths=(16, 32, 64)):
        super().__init__()
        layers = []
        cin = in_ch
        for w in widths:
            layers += [ # convolution layers
                nn.Conv2d(cin, w, kernel_size=3, padding=1),
                nn.BatchNorm2d(w),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1) # flattens it
        self.dropout = nn.Dropout(p=0.3)
        self.head = nn.Linear(widths[-1], num_classes) # classification layer

    def forward(self, x):                  # x: (B, 1, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1) 
        x = self.dropout(x)
        return self.head(x)                # logits
