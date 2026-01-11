---

# README.md

````md
# Deep Residual Learning for Image Recognition (ResNet) — Paper-faithful Cheat Sheet

**Paper**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.  
*Deep Residual Learning for Image Recognition* (arXiv:1512.03385). :contentReference[oaicite:1]{index=1}

이 문서는 ResNet(“v1”/post-activation)의 핵심 정의(식), 블록 구조(그림), 다운샘플링 규칙(표), CIFAR-10 실험 설정(절)을 **논문 기준으로 정리**하고, 그에 대응하는 **PyTorch 레퍼런스 구현**을 함께 제공한다. :contentReference[oaicite:2]{index=2}

---

## Contents
- [1. Motivation: Degradation Problem](#1-motivation-degradation-problem)
- [2. Residual Learning: Eq. (1), Eq. (2)](#2-residual-learning-eq-1-eq-2)
- [3. Building Blocks](#3-building-blocks)
- [4. Shortcut Options (A/B/C)](#4-shortcut-options-abc)
- [5. Network Architectures](#5-network-architectures)
  - [5.1 ImageNet (Table 1)](#51-imagenet-table-1)
  - [5.2 CIFAR-10 (Section 4.2)](#52-cifar-10-section-42)
- [6. Training Recipe from the Paper](#6-training-recipe-from-the-paper)
- [7. Reference PyTorch Implementation](#7-reference-pytorch-implementation)
- [8. Paper → Code Mapping](#8-paper--code-mapping)

---

## 1. Motivation: Degradation Problem

> “with the network depth increasing, accuracy gets saturated … and then degrades rapidly.” :contentReference[oaicite:3]{index=3}

- (KR) 논문은 깊이가 증가할수록 항상 성능이 좋아지지 않으며, 어느 지점 이후에는 **정확도가 포화(saturation) 후 오히려 하락(degradation)**할 수 있음을 관찰한다. :contentReference[oaicite:4]{index=4}

---

## 2. Residual Learning: Eq. (1), Eq. (2)

ResNet의 기본 아이디어는 “직접 목표 함수를 학습”하기보다 “입력에 대한 잔차(residual)를 학습”하도록 블록을 재정의하는 것이다. :contentReference[oaicite:5]{index=5}

### Eq. (1) Identity shortcut
> “y = F(x, {Wi}) + x.” :contentReference[oaicite:6]{index=6}

\[
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
\]

- (KR) 입력 \(\mathbf{x}\)를 그대로 더하는 **identity shortcut**을 사용하면, 블록은 \(\mathcal{F}\) (잔차)만 학습하면 된다. :contentReference[oaicite:7]{index=7}

### Eq. (2) Projection shortcut (dimension mismatch)
> “we can perform a linear projection Ws … to match the dimensions” :contentReference[oaicite:8]{index=8}

\[
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s \mathbf{x}
\]

- (KR) 해상도/채널 수 변경으로 \(\mathbf{x}\)와 \(\mathcal{F}(\mathbf{x})\) 차원이 다르면, shortcut에 **선형 사상(보통 \(1\times1\) conv)** \(W_s\)를 두어 차원을 맞춘다. :contentReference[oaicite:9]{index=9}

### Post-activation (ResNet v1)
> “We adopt the second nonlinearity after the addition.” :contentReference[oaicite:10]{index=10}

- (KR) **덧셈 이후에 ReLU**(두 번째 비선형)를 둔다. (ResNet v1의 전형적인 “post-activation” 형태) :contentReference[oaicite:11]{index=11}

---

## 3. Building Blocks

### 3.1 Basic Block (2-layer residual function)
- 구조: \(3\times3\) conv → BN → ReLU → \(3\times3\) conv → BN → Add(shortcut) → ReLU  
- (KR) Basic block은 “두 개의 3×3 conv로 이루어진 residual function”에 대응한다. :contentReference[oaicite:12]{index=12}

### 3.2 Bottleneck Block (for deeper nets)
- 구조: \(1\times1\) → \(3\times3\) → \(1\times1\) (각 conv 뒤 BN, 마지막 add 뒤 ReLU)  
- (KR) 깊은 모델(50/101/152)에서는 연산량/파라미터 효율을 위해 bottleneck을 사용한다. :contentReference[oaicite:13]{index=13}

---

## 4. Shortcut Options (A/B/C)

논문은 “차원이 증가하는 경우” shortcut을 구성하는 대표 옵션을 비교한다. :contentReference[oaicite:14]{index=14}

- **Option A**: identity shortcut + **zero-padding**(차원 증가 시)  
  > “zero-padding for increasing dimensions (option A).” :contentReference[oaicite:15]{index=15}

- **Option B**: 차원 증가 시 **projection shortcut**, 그 외는 identity  
- **Option C**: 모든 shortcut을 projection으로 구성  
  > “(C) all shortcuts are projections.” :contentReference[oaicite:16]{index=16}

또한 논문은 “A/B/C 모두 plain 대비 개선되며, C는 비용 증가로 이후 실험에선 사용하지 않는다”고 설명한다. :contentReference[oaicite:17]{index=17}  
> “we do not use option C in the rest of this paper” :contentReference[oaicite:18]{index=18}

---

## 5. Network Architectures

### 5.1 ImageNet (Table 1)

- (KR) ResNet-18/34는 basic block, ResNet-50/101/152는 bottleneck을 사용한다. :contentReference[oaicite:19]{index=19}
- 다운샘플링 규칙:
  > “Down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2.” :contentReference[oaicite:20]{index=20}

즉, stage 경계에서 **해당 stage의 첫 블록**이 stride=2로 공간 크기를 줄인다. :contentReference[oaicite:21]{index=21}

### 5.2 CIFAR-10 (Section 4.2)

논문 CIFAR-10 설정은 입력이 32×32이므로 ImageNet stem(7×7, stride2 + maxpool)을 쓰지 않고, 다음과 같은 단순한 형태를 사용한다. :contentReference[oaicite:22]{index=22}

- subsampling: stride=2 conv로 수행 :contentReference[oaicite:23]{index=23}  
- ending: global average pooling + 10-way FC + softmax :contentReference[oaicite:24]{index=24}  
- 총 weighted layers:
  > “There are totally 6n+2 stacked weighted layers.” :contentReference[oaicite:25]{index=25}

논문 표(요약): :contentReference[oaicite:26]{index=26}

| output map size | 32×32 | 16×16 | 8×8 |
|---|---:|---:|---:|
| # layers | 1+2n | 2n | 2n |
| # filters | 16 | 32 | 64 |

그리고 CIFAR-10에서는 shortcut을 다음처럼 사용한다고 명시한다:  
> “we use identity shortcuts in all cases (i.e., option A)” :contentReference[oaicite:27]{index=27}

---

## 6. Training Recipe from the Paper

### 6.1 Implementation detail: BN placement
> “BN … right after each convolution and before activation” :contentReference[oaicite:28]{index=28}

- (KR) Conv 뒤에 BN, 그 다음 ReLU를 둔다. 그리고 add 이후에도 ReLU를 둔다(post-activation). :contentReference[oaicite:29]{index=29}

### 6.2 CIFAR-10 schedule & augmentation (paper)
> “We start with a learning rate of 0.1, divide it by 10 at 32k and 48k iterations … terminate training at 64k.” :contentReference[oaicite:30]{index=30}

> “4 pixels are padded on each side, and a 32×32 crop is randomly sampled … or its horizontal flip.” :contentReference[oaicite:31]{index=31}

- (KR) 배치 128, lr=0.1 시작 → 32k/48k iters에서 1/10 → 64k 종료. :contentReference[oaicite:32]{index=32}  
- (KR) 학습 시: 좌우/상하 4픽셀 패딩 후 랜덤 32×32 crop, horizontal flip. 테스트는 single view만 평가. :contentReference[oaicite:33]{index=33}

---

## 7. Reference PyTorch Implementation

아래 구현은 다음을 “논문 규칙 그대로” 만족한다. :contentReference[oaicite:34]{index=34}  
- Eq.(1)/(2) shortcut: identity 또는 projection(1×1 conv + BN) :contentReference[oaicite:35]{index=35}  
- Conv→BN→ReLU, add 이후 ReLU (post-activation) :contentReference[oaicite:36]{index=36}  
- CIFAR: 3-stage(32/16/8), filters(16/32/64), 6n+2, option A 기본 :contentReference[oaicite:37]{index=37}  
- ImageNet: stage 전환 첫 블록 stride=2 (conv3_1/conv4_1/conv5_1) :contentReference[oaicite:38]{index=38}  

> **Note (implementation detail)**  
> bottleneck에서 “stride=2를 1×1에 둘지 3×3에 둘지”는 프레임워크별 변형이 존재한다.  
> 본 레퍼런스는 “stage의 첫 블록이 stride=2로 downsample한다”는 논문 규칙을 코드에서 명시적으로 보장한다. :contentReference[oaicite:39]{index=39}

---

### 7.1 `model.py`

<details>
<summary><b>Click to expand</b></summary>

```python
from typing import List, Type, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=False)


class ShortcutProjection(nn.Module):
    """Eq.(2): W_s x implemented as 1x1 conv + BN."""
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.op = nn.Sequential(
            conv1x1(in_ch, out_ch, stride=stride),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ShortcutZeroPad(nn.Module):
    """
    Option A (CIFAR): identity shortcut + zero-padding for channel increase,
    stride-2 subsampling when spatial size changes.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        assert out_ch >= in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride != 1:
            x = x[:, :, :: self.stride, :: self.stride]
        ch_pad = self.out_ch - self.in_ch
        if ch_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, ch_pad))
        return x


class BasicBlockV1(nn.Module):
    """
    Basic Block (2x 3x3), post-activation:
    Conv-BN-ReLU, Conv-BN, Add, ReLU
    """
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int, shortcut: str = "projection"):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = conv3x3(out_ch, out_ch, stride=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            if shortcut == "projection":
                self.shortcut = ShortcutProjection(in_ch, out_ch, stride=stride)
            elif shortcut == "zero_pad":
                self.shortcut = ShortcutZeroPad(in_ch, out_ch, stride=stride)
            else:
                raise ValueError(f"Unknown shortcut mode: {shortcut}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out


class BottleneckV1(nn.Module):
    """
    Bottleneck (1x1, 3x3, 1x1), post-activation.
    expansion=4 so out_ch = planes*4.
    """
    expansion = 4

    def __init__(self, in_ch: int, planes: int, stride: int, shortcut: str = "projection"):
        super().__init__()
        out_ch = planes * self.expansion

        # NOTE: stride placement can vary across implementations.
        # Here we place stride on the 3x3 conv (common in modern libraries),
        # while ensuring the stage transition uses stride=2 on the first block.
        self.conv1 = conv1x1(in_ch, planes, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = conv1x1(planes, out_ch, stride=1)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            if shortcut == "projection":
                self.shortcut = ShortcutProjection(in_ch, out_ch, stride=stride)
            elif shortcut == "zero_pad":
                self.shortcut = ShortcutZeroPad(in_ch, out_ch, stride=stride)
            else:
                raise ValueError(f"Unknown shortcut mode: {shortcut}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + identity
        out = self.relu(out)
        return out


class ResNetV1(nn.Module):
    def __init__(
        self,
        block: Type[nn.Module],
        layers: List[int],
        num_classes: int,
        stem: str,
        shortcut: str,
        cifar_base_channels: int = 16,
    ):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

        if stem == "imagenet":
            self.in_ch = 64
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            stage_planes = [64, 128, 256, 512]
        elif stem == "cifar":
            self.in_ch = cifar_base_channels
            self.stem = nn.Sequential(
                conv3x3(3, cifar_base_channels, stride=1),
                nn.BatchNorm2d(cifar_base_channels),
                nn.ReLU(inplace=True),
            )
            stage_planes = [16, 32, 64]
        else:
            raise ValueError("stem must be 'imagenet' or 'cifar'")

        self.layer1 = self._make_layer(block, stage_planes[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, stage_planes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, stage_planes[2], layers[2], stride=2)

        if stem == "imagenet":
            self.layer4 = self._make_layer(block, stage_planes[3], layers[3], stride=2)
            final_ch = stage_planes[3] * getattr(block, "expansion", 1)
        else:
            self.layer4 = None
            final_ch = stage_planes[2] * getattr(block, "expansion", 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_ch, num_classes)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        expansion = getattr(block, "expansion", 1)

        if block is BottleneckV1:
            layers.append(block(self.in_ch, planes, stride=stride, shortcut=self.shortcut))
            self.in_ch = planes * expansion
            for _ in range(1, blocks):
                layers.append(block(self.in_ch, planes, stride=1, shortcut=self.shortcut))
        else:
            out_ch = planes * expansion
            layers.append(block(self.in_ch, out_ch, stride=stride, shortcut=self.shortcut))
            self.in_ch = out_ch
            for _ in range(1, blocks):
                layers.append(block(self.in_ch, out_ch, stride=1, shortcut=self.shortcut))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---- Builders ----

def resnet_imagenet(depth: int, num_classes: int = 1000, shortcut: str = "projection") -> ResNetV1:
    if depth == 18:
        return ResNetV1(BasicBlockV1, [2, 2, 2, 2], num_classes, stem="imagenet", shortcut=shortcut)
    if depth == 34:
        return ResNetV1(BasicBlockV1, [3, 4, 6, 3], num_classes, stem="imagenet", shortcut=shortcut)
    if depth == 50:
        return ResNetV1(BottleneckV1, [3, 4, 6, 3], num_classes, stem="imagenet", shortcut=shortcut)
    if depth == 101:
        return ResNetV1(BottleneckV1, [3, 4, 23, 3], num_classes, stem="imagenet", shortcut=shortcut)
    if depth == 152:
        return ResNetV1(BottleneckV1, [3, 8, 36, 3], num_classes, stem="imagenet", shortcut=shortcut)
    raise ValueError("Unsupported depth")


def resnet_cifar(depth: int, num_classes: int = 10, shortcut: str = "zero_pad") -> ResNetV1:
    # paper CIFAR: depth = 6n + 2
    assert (depth - 2) % 6 == 0, "CIFAR depth must be 6n+2"
    n = (depth - 2) // 6
    return ResNetV1(BasicBlockV1, [n, n, n], num_classes, stem="cifar", shortcut=shortcut)
````

</details>

---

### 7.2 `train_cifar10.py` (paper schedule skeleton)

<details>
<summary><b>Click to expand</b></summary>

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

from model import resnet_cifar


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Examples: 20/32/44/56/110... where depth = 6n+2
    model = resnet_cifar(depth=56, num_classes=10, shortcut="zero_pad").to(device)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    test_tf = T.Compose([T.ToTensor()])

    trainset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=train_tf)
    testset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=test_tf)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    # Paper: divide by 10 at 32k and 48k iters, stop at 64k iters.
    # 50k/128 ≈ 391 iters/epoch → 32k≈82ep, 48k≈123ep, 64k≈164ep
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 123], gamma=0.1)

    for epoch in range(164):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total, test_loss = 0, 0, 0.0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_loss += criterion(logits, y).item() * x.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        scheduler.step()
        print(f"epoch={epoch:03d} acc={100.0*correct/total:.2f} loss={test_loss/total:.4f}")


if __name__ == "__main__":
    main()
```

</details>

---

## 8. Paper → Code Mapping

* Eq.(1) / Eq.(2) → `BasicBlockV1` / `BottleneckV1` 내부의 `out + shortcut(x)` 및 `ShortcutProjection` 
* “BN right after each conv and before activation” → 각 conv 뒤 `BatchNorm2d`, 그 다음 ReLU 
* “second nonlinearity after the addition” → `out = out + identity; out = relu(out)` 
* CIFAR (6n+2, 16/32/64, option A) → `resnet_cifar(depth=6n+2, shortcut="zero_pad")` 
* ImageNet downsampling (conv3_1/4_1/5_1 stride=2) → 각 stage 첫 블록의 `stride=2` 전달 

---

## References

* He et al., *Deep Residual Learning for Image Recognition*, 2015. 

---
