```mermaid
graph LR
    In[Input: 3 x 32 x 32] --> Conv1[Conv2d: 3 -> 32, k3, p1]
    Conv1 --> BN1[BatchNorm2d]
    BN1 --> Act1[LeakyReLU]
    Act1 --> Pool1[MaxPool2d: s2]
    
    Pool1 --> Conv2[Conv2d: 32 -> 128, k3, p1]
    Conv2 --> BN2[BatchNorm2d]
    BN2 --> Act2[LeakyReLU]
    
    Act2 --> Residual[Residual Path]
    Act2 --> MainPath[Main Path]
    
    MainPath --> Pool2[MaxPool2d: s2]
    Pool2 --> Conv3[Conv2d: 128 -> 128, k3, p1]
    Conv3 --> BN3[BatchNorm2d]
    
    Residual --> PoolRes[MaxPool2d: s2]
    PoolRes --> Skip[Skip Conv2d: 128 -> 128, k3, p1]
    
    BN3 --> Add(( + ))
    Skip --> Add
    
    Add --> Act3[LeakyReLU]
    Act3 --> Drop[Dropout: p=0.3]
    Drop --> GAP[AdaptiveAvgPool2d: 1x1]
    GAP --> Flat[Flatten]
    Flat --> FC[Linear: 128 -> num_classes]
    FC --> Out[Output Logits]