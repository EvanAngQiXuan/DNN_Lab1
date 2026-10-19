```mermaid
graph LR
    classDef enc fill:#d4e6f1,stroke:#1b4f72,stroke-width:2px;
    classDef dec fill:#d5f5e3,stroke:#196f3d,stroke-width:2px;
    classDef skip fill:#fcf3cf,stroke:#b7950b,stroke-width:1px,stroke-dasharray: 5 5;
    classDef io fill:#e5e7e9,stroke:#5d6d7e,stroke-width:2px;

    In["Input Image<br/>(3 x 368 x 560)"]:::io --> E1["enc1: Conv2d(3 -> 32, k=3)<br/>ReLU | (32 x 368 x 560)"]:::enc
    
    E1 --> P1["MaxPool2d(2x2)"]:::enc
    P1 --> E2["enc2: Conv2d(32 -> 64, k=3)<br/>ReLU | (64 x 184 x 280)"]:::enc
    
    E2 --> P2["MaxPool2d(2x2)"]:::enc
    P2 --> E3["enc3: Conv2d(64 -> 128, k=3)<br/>ReLU | (128 x 92 x 140)"]:::enc
    
    E3 --> P3["MaxPool2d(2x2)"]:::enc
    P3 --> E4["enc4: Conv2d(128 -> 256, k=3)<br/>Bottleneck | (256 x 46 x 70)"]:::enc

    %% Up-sampling & Decoder 1
    E4 --> U1["Bilinear Up (2x)<br/>(256 x 92 x 140)"]:::dec
    E3 -. "Skip Concat (128 ch)" .-> Cat1["Concat<br/>(384 x 92 x 140)"]:::skip
    U1 --> Cat1
    Cat1 --> D1["dec1: Conv2d(384 -> 128, k=3)<br/>ReLU | (128 x 92 x 140)"]:::dec

    %% Up-sampling & Decoder 2
    D1 --> U2["Bilinear Up (2x)<br/>(128 x 184 x 280)"]:::dec
    E2 -. "Skip Concat (64 ch)" .-> Cat2["Concat<br/>(192 x 184 x 280)"]:::skip
    U2 --> Cat2
    Cat2 --> D2["dec2: Conv2d(192 -> 64, k=3)<br/>ReLU | (64 x 184 x 280)"]:::dec

    %% Up-sampling & Decoder 3
    D2 --> U3["Bilinear Up (2x)<br/>(64 x 368 x 560)"]:::dec
    E1 -. "Skip Concat (32 ch)" .-> Cat3["Concat<br/>(96 x 368 x 560)"]:::skip
    U3 --> Cat3
    Cat3 --> D3["dec3: Conv2d(96 -> 32, k=3)<br/>ReLU | (32 x 368 x 560)"]:::dec

    %% Output
    D3 --> D4["dec4: Conv2d(32 -> 1, k=1)<br/>Output Logits | (1 x 368 x 560)"]:::dec
    D4 --> Out["Segmentation Logits"]:::io