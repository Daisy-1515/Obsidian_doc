# Flash Attention 阅读
## 图片分析
![[Pasted image 20251021205636.png]]
这个图描述的是 FlashAttention 算法与传统计算框架（例如 PyTorch）在计算效率上的对比，以及它如何通过优化内存访问来提升计算性能。我们可以从以下几个方面分析图的内容：
### 1. **内存层次结构（Memory Hierarchy）**
图的上部分展示了 GPU 内存层次结构的带宽和内存大小。不同的内存类型有不同的带宽和大小
- **GPU SRAM**（片上内存）：带宽 19 TB/s，内存大小 20 MB。
- **GPU HBM**（高带宽内存）：带宽 1.5 TB/s，内存大小 40 GB。
- **Main Memory (CPU DRAM)**（主内存/CPU内存）：带宽 12.8 GB/s，内存大小 > 1 TB。
这个层次结构显示了不同内存之间的带宽差异，其中 GPU 的 SRAM 是带宽最高的，而主内存（DRAM）带宽较低。这是优化 FlashAttention 时需要考虑的关键因素，目的是减少数据从较慢内存（如 DRAM）到更快内存（如 HBM 或 SRAM）的移动。
### 2. **FlashAttention 中的计算过程**
图的中间部分展示了 FlashAttention 在处理输入数据时的内存访问模式
- **Q, K, V** 是输入的三个矩阵（查询矩阵 Q、键矩阵 K 和值矩阵 V）。
- 计算过程包含了多个步骤，包括将数据块复制到 **SRAM**、执行矩阵计算等。
- **Outer Loop** 和 **Inner Loop** 表示在计算过程中嵌套的循环，其中 **Outer Loop** 对整个计算进行循环，而 **Inner Loop** 处理单个数据块的计算。
- 数据从 **HBM** 输出，经过一系列内存复制和计算操作，最终得到结果。
这种内存访问模式的优化关键在于减少频繁访问较慢的内存（如 DRAM），而优先使用带宽更高的内存（如 SRAM 和 HBM），以减少 I/O 操作。
### 3. **PyTorch 与 FlashAttention 的对比**
图的右侧展示了 FlashAttention 和 PyTorch 在 GPT-2 上进行 Attention 计算时的时间对比：
- **PyTorch**：使用标准的操作会花费更多时间，特别是在矩阵乘法（Matmul）、Dropout、Softmax、Mask 等步骤上。
- **FlashAttention**：通过优化内存访问和使用融合的内核（Fused Kernel），FlashAttention 显著减少了计算时间（在 GPU 上的执行时间显著降低）。
### 总结：

- **优化目标**：FlashAttention 通过 ==**IO-aware**== 优化，减少了内存访问的开销，特别是在 GPU 内存和 SRAM 之间的读写。
- **显著的性能提升**：FlashAttention 在执行 GPT-2 时显著减少了计算时间，尤其是在处理 Attention 计算时，比 PyTorch 的标准实现更高效。
- **内存优化**：核心的优化策略是利用内存层次结构，通过高效的内存拷贝和计算布局，减少对慢速内存的依赖。



## Standard Attention Backward Pass🧩 中文伪代码（带结构缩进）

```text
输入：
    矩阵 Q, K, V ∈ ℝ^{N×d} 存储在 HBM（高带宽内存）中
    片上 SRAM 大小为 M
    softmax 缩放常数 τ ∈ ℝ
    掩码函数 mask
    dropout 概率 p_drop

过程：

1. 初始化伪随机数生成器状态 R，并保存到 HBM。

2. 设置分块大小：
       B_c = M / (4d)
       B_r = min(M / (4d), d)

3. 初始化下列变量到 HBM：
       O = 0_{N×d}       // 输出矩阵
       l = 0_{N}         // 每行 softmax 的归一化项
       m = 1_{N}         // 每行 softmax 的最大值（数值稳定用）

4. 将 Q 按行划分为 Tr = ⌈N / B_r⌉ 个块：
       Q₁, Q₂, …, Q_Tr，每个大小为 B_r × d。

   将 K, V 按行划分为 Tc = ⌈N / B_c⌉ 个块：
       K₁, K₂, …, K_Tc，V₁, V₂, …, V_Tc，每个大小为 B_c × d。

5. 将 O, l, m 也按相同行块划分：
       O₁…O_Tr, l₁…l_Tr, m₁…m_Tr。

6. 对每个列块 j = 1 … Tc：
       7. 从 HBM 载入 K_j, V_j 到 SRAM。

       8. 对每个行块 i = 1 … Tr：
              9. 从 HBM 载入 Q_i, O_i, l_i, m_i 到 SRAM。

              10. 在片上计算：
                     S_ij = τ × (Q_i × K_jᵀ) ∈ ℝ^{B_r×B_c}

              11. 应用掩码：
                     S_masked_ij = mask(S_ij)

              12. 数值稳定化：
                     m̃_ij = rowmax(S_masked_ij) ∈ ℝ^{B_r}
                     P̃_ij = exp(S_masked_ij - m̃_ij) ∈ ℝ^{B_r×B_c}
                     l̃_ij = rowsum(P̃_ij) ∈ ℝ^{B_r}

              13. 更新稳定化参数：
                     m_new_i = max(m_i, m̃_ij)
                     l_new_i = exp(m_i - m_new_i) * l_i
                                + exp(m̃_ij - m_new_i) * l̃_ij

              14. 应用 dropout：
                     P̃_dropped_ij = dropout(P̃_ij, p_drop)

              15. 更新输出 O：
                     O_i = diag(l_new_i)^{-1} *
                           ( diag(l_i) * exp(m_i - m_new_i) * O_i
                             + exp(m̃_ij - m_new_i) * (P̃_dropped_ij × V_j) )

              16. 将更新后的 O_i, l_i = l_new_i, m_i = m_new_i 写回 HBM。

       17. 结束行块循环 i。

8. 结束列块循环 j。

9. 返回结果：
       O, l, m, R
```

### Backward Pass Pseudocode

```
Require: Matrices Q, K, V ∈ R^N×d in HBM, on-chip SRAM of size M.

1: Set the pseudo-random number generator state to R.
2: Set block sizes Bc = M / (4d), Br = min(M / (4d), d).

3: Divide Q into Tr = ⌈N / Br⌉ blocks Q1, ..., QTr of size Br×d each,
   and divide K, V into Tc = ⌈N / Bc⌉ blocks K1, ..., KTc and V1, ..., VTc, of size Bc×d each.

4: Divide O into Tr blocks O1, ..., OTr of size Br×d each,
   divide dO into Tr blocks dO1, ..., dOTr of size Br×d each,
   divide l into Tr blocks l1, ..., lTr of size Br each,
   divide m into Tr blocks m1, ..., mTr of size Br each.

5: Initialize dQ = 0 ∈ R^N×d in HBM and divide it into Tr blocks dQ1, ..., dQTr of size Br×d each.
   Initialize dK = 0 ∈ R^N×d, dV = 0 ∈ R^N×d in HBM and divide dK, dV into Tc blocks
   dK1, ..., dKTc and dV1, ..., dVTc, of size Bc×d each.

6: For j = 1 to Tc do
7:     Load K_j, V_j from HBM to on-chip SRAM.
8:     Initialize 𝑑̃K_j = 0 ∈ R^{Bc×d}, 𝑑̃V_j = 0 ∈ R^{Bc×d} on SRAM.
9:     For i = 1 to Tr do
10:         Load Q_i, O_i, dO_i, dQ_i, l_i, m_i from HBM to on-chip SRAM.
11:         On-chip, compute S_{ij} = τ Q_i K_j^T ∈ R^{Br×Bc}.
12:         On-chip, apply masking: S_masked_{ij} = mask(S_{ij}).
13:         On-chip, compute P_{ij} = diag(l_i)⁻¹ * exp(S_masked_{ij} - m_i) ∈ R^{Br×Bc}.
14:         On-chip, compute dropout mask Z_{ij} ∈ R^{Br×Bc}, where each entry is
             1/(1 - p_drop) with probability (1 - p_drop), and 0 otherwise.
15:         On-chip, compute P_dropped_{ij} = P_{ij} ⊙ Z_{ij}  (pointwise multiply).
16:         On-chip, update 𝑑̃V_j ← 𝑑̃V_j + P_dropped_{ij}^T dO_i  ∈ R^{Bc×d}.
17:         On-chip, compute dP_dropped_{ij} = dO_i V_j^T ∈ R^{Br×Bc}.
18:         On-chip, compute dP_{ij} = dP_dropped_{ij} ⊙ Z_{ij}.
19:         On-chip, compute D_i = rowsum(dO_i ⊙ O_i) ∈ R^{Br}.
20:         On-chip, compute dS_{ij} = τ * (P_{ij} ⊙ (dP_{ij} - D_i)) ∈ R^{Br×Bc}.
21:         Write dQ_i ← dQ_i + dS_{ij} K_j ∈ R^{Br×d} to HBM.
22:         On-chip, update 𝑑̃K_j ← 𝑑̃K_j + dS_{ij}^T Q_i ∈ R^{Bc×d}.
23:     End for
24:     Write dK_j ← 𝑑̃K_j, dV_j ← 𝑑̃V_j to HBM.
25: End for
26: Return dQ, dK, dV.
```
以下是带有缩进的中文版本：
```
要求：矩阵 Q, K, V ∈ R^N×d 存储在 HBM 中，片上 SRAM 大小为 M。

1: 设置伪随机数生成器的状态为 R。
2: 设置块大小 Bc = M / (4d)，Br = min(M / (4d), d)。

3: 将 Q 分割为 Tr = ⌈N / Br⌉ 个块 Q1, ..., QTr，每个块大小为 Br×d，
   将 K 和 V 分割为 Tc = ⌈N / Bc⌉ 个块 K1, ..., KTc 和 V1, ..., VTc，每个块大小为 Bc×d。

4: 将 O 分割为 Tr 个块 O1, ..., OTr，每个块大小为 Br×d，
   将 dO 分割为 Tr 个块 dO1, ..., dOTr，每个块大小为 Br×d，
   将 l 分割为 Tr 个块 l1, ..., lTr，每个块大小为 Br，
   将 m 分割为 Tr 个块 m1, ..., mTr，每个块大小为 Br。

5: 初始化 dQ = 0 ∈ R^N×d 存储在 HBM 中，并将其分割为 Tr 个块 dQ1, ..., dQTr，每个块大小为 Br×d。
   初始化 dK = 0 ∈ R^N×d，dV = 0 ∈ R^N×d 存储在 HBM 中，并将 dK 和 dV 分割为 Tc 个块
   dK1, ..., dKTc 和 dV1, ..., dVTc，每个块大小为 Bc×d。

6: 对于 j = 1 到 Tc，执行：
    7: 从 HBM 加载 K_j 和 V_j 到片上 SRAM。
    8: 在 SRAM 中初始化 𝑑̃K_j = 0 ∈ R^{Bc×d}，𝑑̃V_j = 0 ∈ R^{Bc×d}。
    9: 对于 i = 1 到 Tr，执行：
        10: 从 HBM 加载 Q_i, O_i, dO_i, dQ_i, l_i, m_i 到片上 SRAM。
        11: 在片上计算 S_{ij} = τ Q_i K_j^T ∈ R^{Br×Bc}。
        12: 在片上应用掩码：S_masked_{ij} = mask(S_{ij})。
        13: 在片上计算 P_{ij} = diag(l_i)⁻¹ * exp(S_masked_{ij} - m_i) ∈ R^{Br×Bc}。
        14: 在片上计算 dropout 掩码 Z_{ij} ∈ R^{Br×Bc}，其中每个条目的值为
            1 / (1 - p_drop) 以概率 (1 - p_drop)，以概率 p_drop 为 0。
        15: 在片上计算 P_dropped_{ij} = P_{ij} ⊙ Z_{ij}（逐点乘法）。
        16: 在片上更新 𝑑̃V_j ← 𝑑̃V_j + P_dropped_{ij}^T dO_i ∈ R^{Bc×d}。
        17: 在片上计算 dP_dropped_{ij} = dO_i V_j^T ∈ R^{Br×Bc}。
        18: 在片上计算 dP_{ij} = dP_dropped_{ij} ⊙ Z_{ij}。
        19: 在片上计算 D_i = rowsum(dO_i ⊙ O_i) ∈ R^{Br}。
        20: 在片上计算 dS_{ij} = τ * (P_{ij} ⊙ (dP_{ij} - D_i)) ∈ R^{Br×Bc}。
        21: 将 dQ_i ← dQ_i + dS_{ij} K_j ∈ R^{Br×d} 写入 HBM。
        22: 在片上更新 𝑑̃K_j ← 𝑑̃K_j + dS_{ij}^T Q_i ∈ R^{Bc×d}。
    23: 结束内层循环。
    24: 将 dK_j ← 𝑑̃K_j，dV_j ← 𝑑̃V_j 写入 HBM。
25: 结束外层循环。

26: 返回 dQ，dK，dV。
```

### 🧠 说明与注解

|符号|含义|
|---|---|
|**HBM**|High Bandwidth Memory，高带宽显存，容量大但访问延迟高。|
|**SRAM**|On-chip Static RAM，片上缓存，容量小但访问速度快。|
|**B_r, B_c**|行、列分块大小，控制每次载入到 SRAM 的数据量。|
|**m_i, l_i**|数值稳定 softmax 的临时变量（分别存储最大值和归一化和）。|
|**mask()**|用于实现 attention mask（如 padding mask 或 causal mask）。|
|**dropout()**|随机丢弃部分权重以防过拟合。|
|**τ**|softmax 缩放常数，通常是 1/√d。|

---



## ==论文如何做softmax分块的：==  

![[Snipaste_2025-10-22_20-48-19.png]]

c ## Softmax 分块计算（Tiling Softmax）公式解析

这个公式的意义在于说明：
先寻找==局部最大值==softmax，再在局部最大值中寻找最大值来完成最终的softmax。

当我们把 **Softmax 的输入向量按块分割（Tiling）** 时，可以 **分块地计算 Softmax**，而不需要一次性加载和处理整个长向量。  
这样做既能保持数值稳定性（避免指数溢出），又能降低内存使用，非常适合像 FlashAttention 这样的高效注意力实现。

---

### 🔹1. 背景：为什么要分块（Tiling）

在标准 Softmax 中，对于一个向量 $x \in \mathbb{R}^N$，我们计算：

$$
\text{Softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

但如果 $N$ 很大（例如 Transformer 里上千甚至上万的 token），直接计算会：

* 需要在显存中存储整个 $x$ 
* 造成指数溢出或数值不稳定

因此我们把 \( x \) **分成多个小块** 来逐步计算，比如  
$x = [x^{(1)}, x^{(2)}]$，每个块的长度为$B$。

### 🔹2. 公式分解说明

假设：

$$
x^{(1)}, x^{(2)} \in \mathbb{R}^B, \quad x = [x^{(1)}, x^{(2)}] \in \mathbb{R}^{2B}
$$

Softmax 的关键中间量定义如下：

$$
m(x) := \max_i x_i, \quad f(x) := e^{x - m(x)}, \quad \ell(x) := \sum_i f(x), \quad \text{Softmax}(x) = \frac{f(x)}{\ell(x)}
$$

---

#### （1）最大值的分解

$$
m(x) = \max(m(x^{(1)}), m(x^{(2)}))
$$

也就是整体的最大值是==每个块最大值中的最大者==。  
👉 这样我们只需要在每个块里找到最大值，然后取全局最大，减少了一次全局扫描。

---

#### （2）指数项的分解

$$
f(x) =
\begin{bmatrix}
e^{m(x^{(1)}) - m(x)} f(x^{(1)}) \\
e^{m(x^{(2)}) - m(x)} f(x^{(2)})
\end{bmatrix}
$$

意思是：  
每个块的$f(x^{(i)}) = e^{x^{(i)} - m(x^{(i)})}$先在局部计算好，  
然后再乘一个修正系数$e^{m(x^{(i)}) - m(x)}$，  
这个修正是因为不同块的最大值不同。

---

#### （3）分母的分解

$$
\ell(x) = e^{m(x^{(1)}) - m(x)} \ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)} \ell(x^{(2)})
$$

即：总的归一化项（分母）是各块的归一化项加权求和。  
权重也是由局部最大值与全局最大值的差控制的。

---

#### （4）最终 Softmax

$$
\text{Softmax}(x) = \frac{f(x)}{\ell(x)}
$$

这一步仍然是标准的 Softmax，只不过分子分母都是通过“分块 + 缩放”组合得到的。

---

### 🔹3. 举个例子

假设：

$$
x^{(1)} = [2, 4, 6], \quad x^{(2)} = [1, 3, 5]
$$

#### Step 1: 求每块最大值

$$
m(x^{(1)}) = 6, \quad m(x^{(2)}) = 5
$$

$$
m(x) = \max(6, 5) = 6
$$

---

#### Step 2: 计算每块的局部 f 值

$$
f(x^{(1)}) = e^{x^{(1)} - 6} = [e^{-4}, e^{-2}, e^{0}]
$$

$$
f(x^{(2)}) = e^{x^{(2)} - 5} = [e^{-4}, e^{-2}, e^{0}]
$$

---

#### Step 3: 修正并拼接

$$
f(x) =
\begin{bmatrix}
e^{6-6} f(x^{(1)}) \\
e^{5-6} f(x^{(2)})
\end{bmatrix}
=
\begin{bmatrix}
f(x^{(1)}) \\
e^{-1} f(x^{(2)})
\end{bmatrix}
$$

---

#### Step 4: 分母

$$
\ell(x) = e^{6-6}\ell(x^{(1)}) + e^{5-6}\ell(x^{(2)}) = \ell(x^{(1)}) + e^{-1}\ell(x^{(2)})
$$

---

#### Step 5: Softmax

$$
\text{Softmax}(x) = \frac{f(x)}{\ell(x)}
$$

---

### 🔹4. 直观理解

* Softmax 需要指数计算，因此数值容易溢出。通过减去最大值 \( m(x) \)，可以让指数项更小、更稳定。
* 分块（tiling）后，不需要一次处理整个长向量，而是按小块分阶段计算：
  * 每块先独立计算；
  * 再组合成整体结果；
  * 所有计算都能在 GPU 的片上缓存（SRAM）中完成，而不是频繁访问慢速显存（HBM）。

---

### ✅ 总结

这个公式体现了 **“Softmax 的分块可组合性”**：

> 你可以对每个块单独计算局部 Softmax 的中间量（最大值、指数和、归一化项），然后用缩放因子组合成全局 Softmax。

这是 FlashAttention 的核心思想之一 ——  
**在不牺牲数值稳定性的前提下，用分块（tiling）降低显存读写成本，实现更快的 Attention 计算。**




## 算法伪代码
```


Require: Matrices Q, K, V ∈ R^N×d in HBM, on-chip SRAM of size M.
1: Set block sizes Bc = M / (4d), Br = min(M / (4d), d).
2: Initialize O = 10^oN×d ∈ R^N×d, l = 10^oN² ∈ R^N×m, m = 1^oN² ∈ R^N×1 in HBM.
3: Divide Q into Tr = l × N / Br × m blocks Q1, Q2, ..., QTr of size Br×d each, and divide K, V into Tc = l × N / Bc × m blocks K1, K2, ..., KTc and V1, V2, ..., VTc, of size Bc×d each.
4: Divide O into Tr blocks O1, O2, ..., OTr of size Br×d each, divide l into Tr blocks l1, l2, ..., lTr of size Br each, divide m into Tr blocks m1, m2, ..., mTr of size Br each.
5: For j = 1 to Tc do
6:   Load K_j, V_j from HBM to on-chip SRAM.
7:   For i = 1 to Tr do
8:     Load Q_i, O_i, l_i, m_i from HBM to on-chip SRAM.
9:     On-chip, compute S_{ij} = Q_i K_j^T ∈ R^{Br×Bc}.
10:    On-chip, compute m̃_{ij} = rowmax(S_{ij}) ∈ R^{Br}, ̃P_{ij} = exp(S_{ij} - m̃_{ij}) ∈ R^{Br×Bc} (pointwise), l̃_{ij} = rowsum( ̃P_{ij}) ∈ R^{Br}.
11:    On-chip, compute mnew_i = max(m_i, m̃_{ij}) ∈ R^{Br}, lnew_i = exp(m_i - mnew_i) l_i - exp(m̃_{ij} - mnew_i) l̃_{ij} ∈ R^{Br}.
12:    Write O_i = diag(lnew_i) * O_i - diag(l_i) * exp(mnew_i) * O_i + exp(m̃_{ij} - mnew_i) * P̃_{ij} * V_j to HBM.
13:    Write l_i = lnew_i, m_i = mnew_i to HBM.
14:  End for
15: End for
16: Return O.

要求：矩阵 Q, K, V ∈ R^N×d 存储在 HBM 中，片上 SRAM 大小为 M。
1: 设置块大小 Bc = M / (4d)，Br = min(M / (4d), d)。
2: 初始化 O = 10^oN×d ∈ R^N×d，l = 10^oN² ∈ R^N×m，m = 1^oN² ∈ R^N×1 存储在 HBM 中。
3: 将 Q 分割为 Tr = l × N / Br × m 个块 Q1, Q2, ..., QTr，每个块的大小为 Br×d；将 K 和 V 分割为 Tc = l × N / Bc × m 个块 K1, K2, ..., Kc 和 V1, V2, ..., Vc，每个块的大小为 Bc×d。
4: 将 O 分割为 Tr 个块 O1, O2, ..., OTr，每个块的大小为 Br×d；将 l 分割为 Tr 个块 l1, l2, ..., lTr，每个块的大小为 Br；将 m 分割为 Tr 个块 m1, m2, ..., mTr，每个块的大小为 Br。
5: 对于 j = 1 到 Tc，执行：
6: 从 HBM 加载 K_j 和 V_j 到片上 SRAM。
7: 对于 i = 1 到 Tr，执行：
8: 从 HBM 加载 Q_i, O_i, l_i, m_i 到片上 SRAM。
9: 在片上计算 S_{ij} = Q_i K_j^T ∈ R^{Br×Bc}。
10: 在片上计算 m̃_{ij} = rowmax(S_{ij}) ∈ R^{Br}， ̃P_{ij} = exp(S_{ij} - m̃_{ij}) ∈ R^{Br×Bc}（逐点计算）， l̃_{ij} = rowsum( ̃P_{ij}) ∈ R^{Br}。
11: 在片上计算 mnew_i = max(m_i, m̃_{ij}) ∈ R^{Br}， lnew_i = exp(m_i - mnew_i) l_i - exp(m̃_{ij} - mnew_i) l̃_{ij} ∈ R^{Br}。
12: 将 O_i 更新为 diag(lnew_i) * O_i - diag(l_i) * exp(mnew_i) * O_i + exp(m̃_{ij} - mnew_i) * P̃_{ij} * V_j 并写入 HBM。
13: 将 l_i 更新为 lnew_i，m_i 更新为 mnew_i，并写入 HBM。
14: 结束内层循环。
15: 结束外层循环。
16: 返回 O。
```

假设我们使用 $N = 4$ 和 $d = 2$ 的矩阵来演示这个过程。我们将通过具体数值演示该伪代码的执行。

### 步骤 1: 设置块大小

假设片上 SRAM 大小 $M$ 已给定。我们首先计算块大小 $B_c$ 和 $B_r$：

$$

B_c = \frac{M}{4d} = \frac{M}{8}, \quad B_r = \min\left(\frac{M}{8}, d\right)

$$

### 步骤 2: 初始化矩阵

假设我们有以下初始化：

* $O = 10^{o} \cdot 4 \times 2$ 矩阵，存储在 HBM 中。

* $l = 10^{o} \cdot 4^2$ 矩阵，存储在 HBM 中。

* $m = 1^{o} \cdot 4^2$ 矩阵，存储在 HBM 中。

矩阵 $Q, K, V$ 都是大小为 $4 \times 2$ 的矩阵，存储在 HBM 中。

### 步骤 3: 分割矩阵

我们将矩阵 $Q$ 分割成 $T_r = \frac{l \times N}{B_r}$ 个块，每个块的大小为 $B_r \times d$。矩阵 $K$ 和 $V$ 分割成 $T_c = \frac{l \times N}{B_c}$ 个块，每个块的大小为 $B_c \times d$。

### 步骤 4: 再次分割其他矩阵

* 将 $O$ 分割成 $T_r$ 个块 $O_1, O_2, ..., O_{T_r}$，每个块大小为 $B_r \times d$。

* 将 $l$ 分割成 $T_r$ 个块 $l_1, l_2, ..., l_{T_r}$，每个块大小为 $B_r$。

* 将 $m$ 分割成 $T_r$ 个块 $m_1, m_2, ..., m_{T_r}$，每个块大小为 $B_r$。

### 步骤 5: 主循环

对于 $j = 1$ 到 $T_c$：

1. 从 HBM 加载 $K_j, V_j$ 到片上 SRAM。

2. 对于 $i = 1$ 到 $T_r$：

3. 从 HBM 加载 $Q_i, O_i, l_i, m_i$ 到片上 SRAM。

4. 在片上计算 $S_{ij} = Q_i K_j^T$（大小为 $B_r \times B_c$）。

5. 在片上计算：

* $m̃_{ij} = \text{rowmax}(S_{ij})$

* $P_{ij} = \exp(S_{ij} - m̃_{ij})$

* $l_{ij} = \text{rowsum}(P_{ij})$   ）

4. 在片上计算：

* $m_{\text{new}}^i = \max(m_i, m̃_{ij})$  当前块的行最大值（用于数值稳定的 softmax）

* $l_{\text{new}}^i = \exp(m_i - m_{\text{new}}^i) l_i - \exp(m̃_{ij} - m_{\text{new}}^i) l_{ij}$  当前块的行归一化因子对应 softmax 的分母

5. 将 $O_i$ 更新并写入 HBM：$O_i = \text{diag}(l_{\text{new}}^i) \cdot O_i - \text{diag}(l_i) \cdot \exp(m_{\text{new}}^i) \cdot O_i + \exp(m̃_{ij} - m_{\text{new}}^i) \cdot P_{ij} \cdot V_j$。

6. 将 $l_i = l_{\text{new}}^i$，$m_i = m_{\text{new}}^i$ 写入 HBM。

### 步骤 6: 返回结果

最终返回更新后的矩阵 $O$。

### 示例数据

假设 $M = 16$（这是片上 SRAM 的大小），我们有以下具体的矩阵和参数：

1. **矩阵 Q, K, V** 是 $4 \times 2$ 的矩阵：

* $Q = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \\ 7 & 8 \end{pmatrix}$

* $K = \begin{pmatrix} 1 & 1 \\ 2 & 2 \\ 3 & 3 \\ 4 & 4 \end{pmatrix}$

* $V = \begin{pmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \\ 7 & 8 \end{pmatrix}$

2. **初始化矩阵 O, l, m**：

* $O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}$

* $l = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}$

* $m = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}$


### 计算细节

#### 步骤 1: 设置块大小

我们假设片上 SRAM 的大小 $M = 16$，并计算块大小 $B_c$ 和 $B_r$：

$$

B_c = \frac{M}{4d} = \frac{16}{4 \times 2} = 2

$$

$$

B_r = \min\left(\frac{M}{4d}, d\right) = \min\left(\frac{16}{8}, 2\right) = 2

$$

因此，我们的块大小是 $B_c = 2$ 和 $B_r = 2$。

#### 步骤 2: 初始化矩阵

我们根据给定的初始化，得到如下矩阵：

* $O = \begin{pmatrix} 0 & 0 \\ 0 & 0 \\ 0 & 0 \\ 0 & 0 \end{pmatrix}$

* $l = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}$

* $m = \begin{pmatrix} 1 \\ 1 \\ 1 \\ 1 \end{pmatrix}$

#### 步骤 3: 分割矩阵

将矩阵 $Q$ 和 $K$，$V$ 按照块大小 $B_c = 2$ 和 $B_r = 2$ 进行分割：

1. **分割 $Q$**：

* 矩阵 $Q$ 是一个 $4 \times 2$ 的矩阵。

* 根据 $B_r = 2$，我们将 $Q$ 分割为 2 个块，每个块大小为 $2 \times 2$，即：

$Q_1 = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$

$Q_2 = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$

2. **分割 $K$ 和 $V$**：

* 矩阵 $K$ 和 $V$ 是 $4 \times 2$ 的矩阵，按相同的方式分割为 2 个块，每个块大小为 $2 \times 2$：

$K_1 = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}$,

$K_2 = \begin{pmatrix} 3 & 3 \\ 4 & 4 \end{pmatrix}$

$V_1 = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$,

$V_2 = \begin{pmatrix} 5 & 6 \\ 7 & 8 \end{pmatrix}$

#### 步骤 4: 继续分割其他矩阵

我们将 $O$，$l$，和 $m$ 也进行分割，每个矩阵根据块大小 $B_r = 2$ 进行分割：

1. **分割 $O$**：

* $O$ 是一个 $4 \times 2$ 的矩阵。

* 由于块大小 $B_r = 2$，我们将 $O$ 分割为 2 个块，每个块大小为 $2 \times 2$：

$O_1 = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$

$O_2 = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$

2. **分割 $l$** 和 **分割 $m$**：

* $l$ 和 $m$ 都是 $4 \times 1$ 的矩阵。

* 每个矩阵会分割为 2 个块：

$l_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$

$l_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$

$m_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$

$m_2 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$

#### 步骤 5: 主循环

接下来，我们进入循环进行计算。首先，对于 $j = 1$ 到 $T_c = 2$，我们加载 $K_j$ 和 $V_j$ 到片上 SRAM。

**对于 $j = 1$**，加载 $K_1$ 和 $V_1$。

然后进入内部循环，对于 $i = 1$ 到 $T_r = 2$，加载 $Q_i, O_i, l_i, m_i$ 到片上 SRAM。

**对于 $i = 1$**，加载 $Q_1$, $O_1$, $l_1$, $m_1$。

1. **计算 $S_{ij} = Q_i K_j^T$**：

$$

S_{11} = Q_1 K_1^T = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix} = \begin{pmatrix} 5 & 5 \\ 11 & 11 \end{pmatrix}

$$

2. **计算 $m̃_{ij} = \text{rowmax}(S_{ij})$**：

* 对于每一行，取最大值：

$$

m̃_{11} = \begin{pmatrix} 5 \\ 11 \end{pmatrix}

$$

3. **计算 $P_{ij} = \exp(S_{ij} - m̃_{ij})$**：

$$

P_{11} = \exp\left(\begin{pmatrix} 5 & 5 \\ 11 & 11 \end{pmatrix} - \begin{pmatrix} 5 \\ 11 \end{pmatrix}\right) = \exp\left(\begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}\right) = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}

$$

4. **计算 $l_{ij} = \text{rowsum}(P_{ij})$**：

$$

l_{11} = \text{rowsum}\left(\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\right) = \begin{pmatrix} 2 \\ 2 \end{pmatrix}

$$

5. **计算 $m_{\text{new}}^i$ 和 $l_{\text{new}}^i$**：

* 计算 $m_{\text{new}}^1 = \max(m_1, m̃_{11}) = \max(1, 5) = 5$

* 计算 $l_{\text{new}}^1 = \exp(m_1 - m_{\text{new}}^1) l_1 - \exp(m̃_{11} - m_{\text{new}}^1) l_{11}$

$$

l_{\text{new}}^1 = \exp(1 - 5) \begin{pmatrix} 1 \\ 1 \end{pmatrix} - \exp\left(\begin{pmatrix} 5 \\ 11 \end{pmatrix} - 5\right) \begin{pmatrix} 2 \\ 2 \end{pmatrix}

$$

6. **更新 $O_1$**： ==初始化为0==

* 将 $O_1$ 更新为 $O_1 = \text{diag}(l_{\text{new}}^1) \cdot O_1 - \text{diag}(l_1) \cdot \exp(m_{\text{new}}^1) \cdot O_1 + \exp(m̃_{11} - m_{\text{new}}^1) \cdot P_{11} \cdot V_1$。


 ① $\text{diag}(l_{\text{new}}^i) \cdot O_i$  

> 把之前的输出 $O_i$ 乘上新的归一化比例 $l_{\text{new}}^i$。  ==初始化为1==

表示我们正在对之前的结果进行重新缩放，以适配更新后的 softmax 分母。

② $- \text{diag}(l_i) \cdot e^{m_{\text{new}}^i} \cdot O_i$

> 从旧的加权输出中减去多余的部分（由旧的缩放比例 $l_i$、旧的最大值 $m_i$ 决定）。

这一步实现了**数值稳定的 softmax 递推**，避免了指数溢出。

通过比较新旧最大值 $m_{\text{new}}^i$ 和 $m_i$，将先前的结果重新对齐到新的指数基准。
 ③ $+ e^{\tilde{m}_{ij} - m_{\text{new}}^i} \cdot \tilde{P}_{ij} \cdot V_j$

> 加上当前块 $j$ 的注意力贡献。

* $\tilde{P}_{ij} = \exp(S_{ij} - \tilde{m}_{ij})$：当前块未归一化的 softmax 权重；

* $V_j$：对应的值向量；

* 指数项 $e^{\tilde{m}_{ij} - m_{\text{new}}^i}$：用于将该块的权重与新的归一化基准对齐。

这项实际相当于计算：

$$

\text{softmax\_increment} = \text{attention}(Q_i, K_j, V_j)

$$

并把它累加进 $O_i$。

6. **写入更新后的 $l_1$ 和 $m_1$** 到 HBM。

#### 步骤 6: 完成

继续执行这个过程直到所有的 $i$ 和 $j$ 循环结束。最终返回更新后的矩阵 $O$。

---

这个步骤展示了如何根据 $N = 4$ 和 $d = 2$ 的设置进行矩阵的分割、计算和更新。













下面链接是FlashAttention算法实例演示
https://blog.csdn.net/HaoBBNuanMM/article/details/135415355 
![[算法分析FlashAttention算法实例演示_flash attention实例-CSDN博客.mp4]]


[[深度学习中的常见操作：Elementwise 与 Reduction 操作解析]]

[[BERT-large]]

[[操作融合（Operation Fusion)]]

==[[Flash Attention 阅读]]==

[[注意力机制中QKV反向传播公式推导]]

下面链接是开源代码
https://github.com/Dao-AILab/flash-attention

---
## 1. **Tiling（分块技术）**

**Tiling**（分块技术）是一种在计算中通过将大矩阵或张量划分为较小的“块”来优化计算性能的技术。通过将数据分成更小的部分，可以更好地利用计算资源，减少内存访问瓶颈，并提高缓存的使用效率。
#### 如何工作：

在计算矩阵乘法或其他操作时，传统方法可能会直接计算整个矩阵，导致大量的内存访问和计算开销。通过将矩阵划分为多个较小的块，计算可以在较小的内存区域内进行，从而提高效率，特别是在 GPU 和其他高性能硬件上，能更好地利用 **高带宽内存**（HBM）和 **片上缓存**（SRAM）。
#### 举例：

假设我们有两个矩阵 AA 和 BB，它们的维度分别是 4×4 ，我们需要计算它们的矩阵乘积 C=A×B
$$

A = \begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}



B = \begin{bmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8 \\
9 & 10 & 11 & 12 \\
13 & 14 & 15 & 16
\end{bmatrix}
$$
直接计算矩阵乘法可能会涉及对整个矩阵的内存访问，导致效率较低。**Tiling** 技术可以将这两个矩阵分成更小的块来计算。例如，假设我们将 AA 和 BB 分成 2×2 的块：
![[Snipaste_2025-10-21_21-11-36.png]]


然后，计算这些小块之间的乘积，而不是直接计算整个矩阵。这样可以更高效地进行内存访问和计算。



## 2. **Recomputation（重新计算）**

**Recomputation**（重新计算）是一种通过在需要时重新计算某些中间结果，而不是存储它们来节省内存的技术。特别是在深度学习中，模型的训练通常会涉及大量的中间结果和临时存储。通过重新计算一些中间结果而不是缓存它们，可以有效减少内存消耗，尤其是在内存受限的环境下。
#### 如何工作：

在训练神经网络时，通常会计算许多中间激活（比如层的输出）并将它们存储以便后续使用。但是如果内存有限，存储所有中间结果可能会导致内存不足或性能瓶颈。**Recomputation** 通过“丢弃”一些中间结果，并在需要时重新计算这些结果，减少内存消耗。
#### 举例：
假设我们在进行 **矩阵乘法** 计算时，计算出中间结果 A×B（其中 A 和 B 是矩阵）。我们可以选择将这个结果存储下来以便后续使用，但如果内存有限，可以选择在每次需要时重新计算它。
1. **不使用 recomputation**：首先存储 A×B 的结果，然后直接用它进行后续计算。
    
    C=A×B
    
    然后直接使用 C 进行其他操作，但这需要存储 C，增加内存消耗。
    
2. **使用 recomputation**：我们可以在每次需要 C 时重新计算 A×B 而不是存储它。
    
    C=A×B(每次需要时重新计算)
    
    这样虽然增加了计算开销，但节省了内存。
    

#### 适用场景：

- **深度学习中的反向传播**：重新计算中间激活值，而不是存储所有中间激活，有助于节省内存，尤其是在训练深度神经网络时。
    
- **大型矩阵运算**：对于需要大量内存的矩阵计算，重新计算中间结果可以减少内存需求。
    

### 总结：

|技术|描述|优势|举例|
|---|---|---|---|
|**Tiling**|将矩阵或张量分成较小的块来进行计算，以提高缓存和内存的利用效率。|更高效的内存访问，减少内存带宽瓶颈，适用于大规模计算。|计算矩阵乘法时，将大矩阵分块后进行局部计算，减少内存访问延迟。|
|**Recomputation**|在需要时重新计算中间结果，而不是存储它们，减少内存消耗。|减少内存占用，适用于内存受限的场景，但会增加计算开销。|训练深度学习模型时，丢弃中间激活值，反向传播时重新计算。|

这两种技术在深度学习和其他大规模计算中都非常重要，尤其是在处理内存限制或计算资源有限的情况下。

However, FlashAttention is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only 25-40% of the theoretical maximum FLOPs/s. We observe that the inefficiency is due to suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes
然而，FlashAttention 的速度仍远不及优化矩阵乘法（GEMM）运算，仅达到理论最大 FLOPs/s 的 25-40%。我们发现，效率低下的原因是 GPU 上不同线程块和翘曲之间的工作分割不够理想，导致低占用率或不必要的共享内存读/写。为了解决这些问题，我们提出了具有更好工作分区的 FlashAttention-2。
In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory.
具体来说，我们
（1）调整算法以减少非 Matmul FLOPs 的数量（==主要是softmax==）
（2）在不同线程块之间并行处理注意力计算（即使是单头计算），以提高占用率，以及
（3）在每个线程块内，在 warps 之间分配工作，以减少通过共享内存的通信。

# FlashAttention-2
##  算法 1 FlashAttention-2 前向传递
### 中文版本（带缩进）

1. 将 $Q$ 分成 $T_r = \left\lceil \frac{N}{B_r} \right\rceil$ 个块 $Q_1, Q_2, \dots, Q_{T_r}$，每个块的大小为 $B_r \times d$，并将 $K$ 和 $V$ 分成 $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ 个块 $K_1, K_2, \dots, K_{T_c}$ 和 $V_1, V_2, \dots, V_{T_c}$，每个块的大小为 $B_c \times d$。
2. 将输出 $O \in \mathbb{R}^{N \times d}$ 分成 $T_r$ 个块 $O_1, O_2, \dots, O_{T_r}$，每个块的大小为 $B_r \times d$，并将 logsumexp $L$ 分成 $T_r$ 个块 $L_1, L_2, \dots, L_{T_r}$，每个块的大小为 $B_r$。
3. 对于 $1 \leq i \leq T_r$，执行以下操作：
* 加载 $Q_i$ 从 HBM 到片上 SRAM。
* 在片上初始化 $O^{(0)}_i = (0)_{B_r \times d} \in \mathbb{R}^{B_r \times d}$，$l^{(0)}_i = (0)_{B_r} \in \mathbb{R}^{B_r}$，$m^{(0)}_i = (-\infty)_{B_r} \in \mathbb{R}^{B_r}$。
* 对于 $1 \leq j \leq T_c$，执行以下操作：
* ==加载 $K_j, V_j$ 从 HBM 到片上 SRAM==。
* 在片上计算 $S^{(j)}_i = Q_i K_j^T \in \mathbb{R}^{B_r \times B_c}$。
* 在片上计算 $m^{(j)}_i = \max(m^{(j-1)}_i, \text{rowmax}(S^{(j)}_i)) \in \mathbb{R}^{B_r}$，$P^{(j)}_i = \exp(S^{(j)}_i - m^{(j)}_i) \in \mathbb{R}^{B_r \times B_c}$（按元素计算），$l^{(j)}_i = \exp(m^{(j-1)}_i - m^{(j)}_i) l^{(j-1)}_i + \text{rowsum}(P^{(j)}_i) \in \mathbb{R}^{B_r}$。
* 在片上计算 $O^{(j)}_i = \text{diag}(\exp(m^{(j-1)}_i - m^{(j)}_i))^{-1} O^{(j-1)}_i + P^{(j)}_i V_j$。
* 结束循环。
1. 在片上计算 $O_i = \text{diag}(l^{(T_c)}_i)^{-1} O^{(T_c)}_i$。
2. 在片上计算 $L_i = m^{(T_c)}_i + \log(l^{(T_c)}_i)$。
3. 将 $O_i$ 写入 HBM 作为第 $i$ 个块的输出 $O$。
4. 将 $L_i$ 写入 HBM 作为第 $i$ 个块的 logsumexp $L$。
5. 结束循环。
6. 返回输出 $O$ 和 logsumexp $L$。


**与1的区别：**
1. load顺序有区别，以前是一列一列计算，现在是一行一行算S。
原本:
![[Snipaste_2025-10-25_17-01-24.png]]
现版:
对于 $1 \leq i \leq T_r$，执行以下操作：
* 加载 $Q_i$ 从 HBM 到片上 SRAM。
* 在片上初始化 $O^{(0)}_i = (0)_{B_r \times d} \in \mathbb{R}^{B_r \times d}$，$l^{(0)}_i = (0)_{B_r} \in \mathbb{R}^{B_r}$，$m^{(0)}_i = (-\infty)_{B_r} \in \mathbb{R}^{B_r}$。
* 对于 $1 \leq j \leq T_c$，执行以下操作：
* ==加载 $K_j, V_j$ 从 HBM 到片上 SRAM==。

### English Version (Indented)

1. Divide $Q$ into $T_r = \left\lceil \frac{N}{B_r} \right\rceil$ blocks $Q_1, Q_2, \dots, Q_{T_r}$, each of size $B_r \times d$, and divide $K$ and $V$ into $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ blocks $K_1, K_2, \dots, K_{T_c}$ and $V_1, V_2, \dots, V_{T_c}$, each of size $B_c \times d$.
2. Divide the output $O \in \mathbb{R}^{N \times d}$ into $T_r$ blocks $O_1, O_2, \dots, O_{T_r}$, each of size $B_r \times d$, and divide the logsumexp $L$ into $T_r$ blocks $L_1, L_2, \dots, L_{T_r}$, each of size $B_r$.
3. For $1 \leq i \leq T_r$, do the following:
* Load $Q_i$ from HBM to on-chip SRAM.
* On-chip, initialize $O^{(0)}_i = (0)_{B_r \times d} \in \mathbb{R}^{B_r \times d}$, $l^{(0)}_i = (0)_{B_r} \in \mathbb{R}^{B_r}$, $m^{(0)}_i = (-\infty)_{B_r} \in \mathbb{R}^{B_r}$.
* For $1 \leq j \leq T_c$, do the following:
* Load $K_j, V_j$ from HBM to on-chip SRAM.
* On-chip, compute $S^{(j)}_i = Q_i K_j^T \in \mathbb{R}^{B_r \times B_c}$.
* On-chip, compute $m^{(j)}_i = \max(m^{(j-1)}_i, \text{rowmax}(S^{(j)}_i)) \in \mathbb{R}^{B_r}$, $P^{(j)}_i = \exp(S^{(j)}_i - m^{(j)}_i) \in \mathbb{R}^{B_r \times B_c}$ (pointwise), $l^{(j)}_i = \exp(m^{(j-1)}_i - m^{(j)}_i) l^{(j-1)}_i + \text{rowsum}(P^{(j)}_i) \in \mathbb{R}^{B_r}$.
* On-chip, compute $O^{(j)}_i = \text{diag}(\exp(m^{(j-1)}_i - m^{(j)}_i))^{-1} O^{(j-1)}_i + P^{(j)}_i V_j$.
* End of loop.
1. On-chip, compute $O_i = \text{diag}(l^{(T_c)}_i)^{-1} O^{(T_c)}_i$.
2. On-chip, compute $L_i = m^{(T_c)}_i + \log(l^{(T_c)}_i)$.
3. Write $O_i$ to HBM as the $i$-th block of output $O$.
4. Write $L_i$ to HBM as the $i$-th block of logsumexp $L$.
5. End of loop.
6. Return the output $O$ and logsumexp $L$.


## [[CUDA中的atomic原子操作]]
## warps中 split K方案 
在warp中，头为1，单头注意力中矩阵乘法，如下图所示，在版本1中，最后O的输出就需要线程最后加上去
$$ 
O_1 = 
  \vec{S_1^{(1)}} \cdot \vec{V^{(1)}} 
+ \vec{S_1^{(2)}} \cdot \vec{V^{(2)}} 
+ \vec{S_1^{(3)}} \cdot \vec{V^{(3)}}
+ \vec{S_1^{(4)}} \cdot \vec{V^{(4)}}
$$
上标表示哪个warp 下标表示第几行
$$ 
O_2 = 
  \vec{S_2^{(1)}} \cdot \vec{V^{(1)}} 
+ \vec{S_2^{(2)}} \cdot \vec{V^{(2)}} 
+ \vec{S_2^{(3)}} \cdot \vec{V^{(3)}}
+ \vec{S_2^{(4)}} \cdot \vec{V^{(4)}}
$$
![[Snipaste_2025-10-29_13-47-37.png]]
![[Pasted image 20251029131824.png]]
![[Pasted image 20251029131828.png]]
## 与Flash Attention-1的区别
1. 在前向传播算法中，**1**把Q,V放在了**内循环**中，计算顺序是计算出所有向量的一部分，即按列从上往下运行；但是在**2**中，是把K，V放在了内循环中，计算顺序是，先计算一部分向量的全部，即从左右往右运算
	**为什么这么做**：
	* 2优化了softmax操作，不想要额外m和l了，就一次性把向量算完了在去算其他的向量，就可以在一次内循环中找到最大值，而不是像1一样要在外循环最后一次才找到所有向量的最大值。相比较下，原本需要O(N)的存储空间就只需要O(N/分块次数)的存储空间来存softmax的缩放系数了。
	* 
   
2. 
## 对Flash Attention评价
在实际的GPU情况过于理想，Flash Attention 2继续优化。

[[end2end名词解释]]


# Flash Attention-3

https://tridao.me/blog/2024/flash3/
