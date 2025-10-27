However, FlashAttention is still not nearly as fast as optimized matrix-multiply (GEMM) operations, reaching only 25-40% of the theoretical maximum FLOPs/s. We observe that the inefficiency is due to suboptimal work partitioning between different thread blocks and warps on the GPU, causing either low-occupancy or unnecessary shared memory reads/writes
然而，FlashAttention 的速度仍远不及优化矩阵乘法（GEMM）运算，仅达到理论最大 FLOPs/s 的 25-40%。我们发现，效率低下的原因是 GPU 上不同线程块和翘曲之间的工作分割不够理想，导致低占用率或不必要的共享内存读/写。为了解决这些问题，我们提出了具有更好工作分区的 FlashAttention-2。
In particular, we (1) tweak the algorithm to reduce the number of non-matmul FLOPs (2) parallelize the attention computation, even for a single head, across different thread blocks to increase occupancy, and (3) within each thread block, distribute the work between warps to reduce communication through shared memory.
具体来说，我们
（1）调整算法以减少非 Matmul FLOPs 的数量（==主要是softmax==）
（2）在不同线程块之间并行处理注意力计算（即使是单头计算），以提高占用率，以及
（3）在每个线程块内，在 warps 之间分配工作，以减少通过共享内存的通信。


#  算法 1 FlashAttention-2 前向传递
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
# 对Flash Attention评价
在实际的GPU情况过于理想，Flash Attention 2继续优化。
[[Flash Attention 阅读]]
[[end2end名词解释]]