좋아. 아래는 **내가 생각하는 “README 하나로 끝내는 최상치 버전”**이야.
👉 **labml 스타일의 논리 흐름**을 유지하면서
👉 **GitHub에서 바로 관리 가능한 Markdown**이고
👉 **수식/용어는 영어, 설명은 한국어 병기**
👉 **논문 → 코드 → CIFAR-10 실습**이 자연스럽게 이어지도록 설계했어.

이걸 **그대로 `README.md`로 두고**, 마음에 안 드는 부분만 깎아나가면 돼.

---

# Deep Residual Learning for Image Recognition (ResNet) — Cheatsheet

> **Paper**: *Deep Residual Learning for Image Recognition*
> **Framework**: PyTorch
> **Architecture**: ResNet v1 (post-activation)
> **Goal**: 논문과 1:1로 매핑되는 reference 구현 + CIFAR-10 실습으로의 자연스러운 확장

---

## Why this document?

This repository is a **paper-faithful ResNet v1 cheatsheet**.

* 수식 → 개념 → 코드가 **한 줄씩 대응**되도록 구성
* ImageNet 논문 구조를 기준으로 설명한 뒤,
* **같은 block을 유지한 채 CIFAR-10 실습으로 확장**

> **KR 요약**
> “ResNet 논문을 읽다가 구현으로 옮길 때,
> ‘어디가 논문에서 나온 거고, 어디가 실전 변형인지’ 헷갈리지 않게 만드는 문서”

---

## 1. Degradation Problem

Deep neural networks suffer from the **degradation problem**:

> As network depth increases, training error **increases**, even though the model is strictly more expressive.

* **KR 설명**
  단순히 layer를 더 쌓는다고 성능이 좋아지지 않는다.
  오히려 깊어질수록 **학습 자체가 어려워지고 정확도가 떨어지는 현상**이 발생한다.

The paper argues that **deeper models should not perform worse** than shallower ones,
because extra layers could simply learn an **identity mapping**.

---

## 2. Residual Learning

Instead of directly learning a mapping ( \mathcal{H}(x) ),
ResNet learns a **residual function**:

[
\mathcal{F}(x) = \mathcal{H}(x) - x
]

so that the original mapping becomes:

[
\mathcal{H}(x) = \mathcal{F}(x) + x
]

* **KR 직관**
  “완전히 새 함수를 학습”하는 대신
  **입력과의 차이(residual)** 만 학습하게 하면 훨씬 쉽다.
* Identity mapping은 ( \mathcal{F}(x) = 0 ) 만 학습하면 된다.

---

## 3. Projection Shortcut

When the shapes of ( \mathcal{F}(x) ) and ( x ) differ
(spatial size or number of channels):

[
\mathcal{H}(x) = \mathcal{F}(x) + W_s x
]

* **KR 설명**

  * feature map의 **H×W** 또는 **C** 가 다르면 단순히 더할 수 없다.
  * 이때 논문은 **learned linear projection** (W_s) 를 제안한다.
* Zero-padding보다 **projection이 더 좋은 성능**을 보였다고 보고함.

---

## 4. Code Mapping — Projection Shortcut

```python
class ShortcutProjection(nn.Module):
    """
    Linear projection: W_s x
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=stride, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))
```

* **논문 대응**

  * ( W_s x )
* **KR 포인트**

  * stride를 main path와 동일하게 적용 → spatial size 일치
  * BN은 논문에서 권장됨

---

## 5. ResNet v1 Block (Post-activation)

### BasicBlock (ResNet-18 / 34)

**Pattern (v1)**

```
Conv → BN → ReLU
Conv → BN
+ shortcut
ReLU
```

```python
class BasicBlockV1(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = ShortcutProjection(in_ch, out_ch, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        s = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + s)
```

* **KR 핵심**

  * shortcut 조건은 **stride 변경 OR 채널 변경**
  * 덧셈 이후 ReLU → **ResNet v1 (post-activation)**

---

## 6. Bottleneck Block (ResNet-50 / 101 / 152)

**Pattern (traditional v1)**

```
1×1 → BN → ReLU
3×3 → BN → ReLU   (stride here)
1×1 → BN
+ shortcut
ReLU
```

* **KR 설명**

  * 3×3 연산을 **bottleneck channel**에서 수행
  * 계산량 감소 + 깊은 네트워크 가능
* stride를 3×3에 주는 방식은 **전통적(논문 계열) 구현**

---

## 7. Stage & Downsampling Rule (중요)

**Paper rule (Table-style)**

* Downsampling (`stride=2`) happens **only at the first block of a new stage**
* Except for the **first stage**, which uses `stride=1`

| Stage  | Common name | First block stride |
| ------ | ----------- | ------------------ |
| layer1 | conv2_x     | 1                  |
| layer2 | conv3_x     | 2                  |
| layer3 | conv4_x     | 2                  |
| layer4 | conv5_x     | 2                  |

* **KR 주의**

  * `len(blocks)` 같은 누적 블록 기준은 논문 규칙을 구현할 수 없다.
  * 반드시 **stage index 기준**으로 stride를 결정해야 한다.

---

## 8. ImageNet Stem (Paper-faithful)

```python
class StemImageNetV1(nn.Module):
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 7, stride=2, padding=3, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return self.pool(x)
```

* **논문 Table 1 그대로**
* ImageNet 입력(224×224)에 최적화된 stem

---

## 9. From Paper to CIFAR-10

논문 구조를 그대로 CIFAR-10(32×32)에 쓰면 **너무 빨리 다운샘플링**된다.

그래서 실습에서는 보통:

### CIFAR Stem

```python
class StemCIFAR(nn.Module):
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
```

* **KR 요약**

  * block / shortcut / stage 규칙은 그대로 유지
  * **stem과 classifier만 교체**

---

## 10. Global Average Pooling

Instead of fixed-size pooling:

```python
nn.AdaptiveAvgPool2d(1)
```

* **KR 설명**

  * 입력 크기에 독립적
  * CIFAR / ImageNet 모두 동일 코드 사용 가능

---

## 11. CIFAR-10 Training Skeleton

> 치트시트 목적이므로 핵심만 정리

* **Data augmentation**

  * `RandomCrop(32, padding=4)`
  * `RandomHorizontalFlip()`
* **Optimizer**

  * SGD(lr=0.1, momentum=0.9, weight_decay=2e-4)
* **Loss**

  * CrossEntropyLoss

---

## Final Notes

* This README follows **ResNet v1 (He et al., 2015)** faithfully.
* CIFAR-10 implementation is a **controlled deviation**, not a different model.
* If you understand every section here,
  you can **read the paper and write the code without guessing**.

---

### Suggested next steps

* Split this README into `docs/` if it grows
* Add stage-wise feature map size tables
* Add ResNet-18 / 50 config presets

---

원하면 다음으로:

* 🔹 **이 README 기준으로 “불필요한 부분 삭제 버전”**
* 🔹 **stage별 shape 표 추가**
* 🔹 **A/B 코드와 정확히 매칭되는 주석 diff**

중 아무거나 바로 이어서 해줄게.
