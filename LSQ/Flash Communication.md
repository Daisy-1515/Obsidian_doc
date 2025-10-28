![[Pasted image 20251027144345.png]]
exp：
1. **RMSNorm**：这是一个归一化操作，应用于输入张量（QKV——查询、键、值）。它为模型中的后续层做准备。
2. **QKV和SelfAttention**：这部分表示用于自注意力机制的查询、键和值张量。SelfAttention模块执行注意力计算，这是像LLaMA这类基于Transformer的模型的核心。
3. **Flash All-Reduce**：这是图中展示的核心创新。Flash All-Reduce是一种高效的分布式计算通信技术，旨在加速张量并行。图中强调了在前向传播过程中使用这一技术，其中注意力机制（QKV）的结果被聚合并在设备间通信。使用低位量化（例如INT4、INT6）有助于减少传输的数据量，从而提高速度，同时不牺牲太多准确性。
4. **Gate with Swish**：这个模块可能指的是一种门控机制，可能与激活函数Swish相关，Swish用于在模型中引入非线性。
5. **Down和Up Projections**：这些指的是在Transformer模型的前馈网络中应用于张量值的转换操作。
总之，这个图表概述了张量如何通过不同的操作，如归一化、自注意力和投影，进行处理，重点介绍了如何通过Flash All-Reduce优化通信，这减少了延迟并提高了分布式模型推理中的效率。这对于大规模模型至关重要，因为通信瓶颈常常成为限制因素。


#  **Flash All-Reduce** 算法：

```text
Algorithm 1: Flash All-Reduce

Input: 
    Communication volume M
    World size N
    Chunk size C
    Quantization bit-width b
    Group size g

Output: 
    Reduced sum S_dq

1. Divide M into T = ⌈ M / C ⌉ chunks.
2. for 1 ≤ i ≤ T do
    3. // Quantize volume to obtain zeros and scales
    4. M_iq, z_i, s_i = FinegrainedQuantize(M_i, b, g);
    
    5. // Each device sends and receives volume from others
    6. All2All(M_iq, z_i, s_i, N);

    7. for 1 ≤ j ≤ N do
        8. M_ij_dq = Dequantize(M_ij_q, z_ij, sij);
    8. end for

    9. S_i = ReduceSum(M_i0_dq, M_i1_dq, ..., M_iN_dq);

    10. S_iq, z_is, s_is = FinegrainedQuantize(S_i, b, g);

    11. // Each device collects the reduced sum from others
    12. All-Gather(S_iq, z_is, s_is, N);

    13. for 1 ≤ j ≤ N do
        15. S_ij_dq = Dequantize(S_iq, z_is, s_is);
    14. end for
3. end for
```
### **算法步骤简要说明**：

1. **数据划分（Step 1）**：将总的通信体积 **M** 划分为 **T** 个数据块，数据块的大小为 **C**。
2. **量化处理（Step 4）**：对每个数据块进行量化，得到量化后的数据、零点（z_i）和缩放因子（s_i）。
3. **数据交换（Step 6）**：使用 **All2All** 操作进行数据交换，确保每个节点都能接收到其他节点的数据。
4. **反量化（Step 8）**：每个节点在接收到其他节点的数据后，进行反量化操作，恢复为原始数据形式。
5. **规约（Step 10）**：对接收到的数据进行规约操作（如求和），得到局部的规约结果 **S_i**。
6. **结果量化（Step 11）**：对规约结果 **S_i** 进行量化，得到量化后的结果。
7. **全收集（Step 13）**：使用 **All-Gather** 操作收集所有节点的规约结果，确保每个节点都有完整的结果。
8. **反量化（Step 15）**：对最终收集的规约结果进行反量化，恢复为浮点数数据形式。

# Thread mapping of fast fine-grained quantization快速细粒度量化的线程映射
![[Pasted image 20251027175102.png]]
8192=8 * 32 * 32 一个block的数据量化
**层次架构：**
*  **chunk -> block -> warp -> thread**
# two-step All-Reduce

![[Snipaste_2025-10-27_17-46-48.png]]

# Comparison of Ring All-Reduce vs. Flash All-Reduce 环形全还原与闪存全还原的比较

| Method                       | Ring All-Reduce          | Flash All-Reduce         |
| ---------------------------- | ------------------------ | ------------------------ |
| **Total Volume**             | $$ \frac{2M (N-1)}{N} $$ | $$ \frac{2M (N-1)}{N} $$ |
| **Reduce Step**              | $$ N - 1 $$              | $$ 1 $$                  |
| **Reduce-Scatter**<br>这个是数据量 | $$ \frac{M}{N} $$        | $$ \frac{M (N-1)}{N} $$  |
| **Gather Step**              | $$ N - 1 $$              | $$ 1 $$                  |
| **All-Gather**<br>这个是数据量     | $$ \frac{M}{N} $$        | $$ \frac{M (N-1)}{N} $$  |
| **QDQ Step**                 | $$ N $$                  | $$ 2 $$                  |

疑问：
==**Reduce Step**怎么变成1了，==
* 因为原本应该取得的数据已经通过ALL2ALL通信交换到了，只需要在GPU内部做求和操作就能取得，所以在通信的操作次数为1
![[Snipaste_2025-10-28_10-21-55.png]]
==**Gather Step**怎么变成1了==，
* 因为架构不需要遵守ring环形架构了，直接使用指令一次性全部同步，和下面图片一样
![[Snipaste_2025-10-28_10-19-32.png]]
这张表格对比了 **Ring All-Reduce** 和 **Flash All-Reduce** 两种通信方法，并列出了它们在 **总通信体积**、**规约步骤**、**规约-分散步骤**（Reduce-Scatter）、**收集步骤**（Gather Step）、**全收集步骤**（All-Gather）、和 **量化-反量化步骤**（QDQ Step）等方面的区别。

下面是表格中术语的详细解释：
### **1. Total Volume**
* **总通信体积**：表示在执行通信时，总数据量的大小。这个值与每个节点的发送和接收数据量有关。
* **Ring All-Reduce**：通信体积为 $2M (N-1) / N$，其中：
* **M** 是每个节点的通信数据量。
* **N** 是参与计算的节点数。
* **2M (N-1)** 表示由于数据在多个节点间传递，因此通信量在规约和同步时是多次的。
* **Flash All-Reduce**：通信体积为 $2M (N-1) / N$，与 Ring All-Reduce 相同，但是通过优化减少了数据传输的复杂度和次数。
### **2. Reduce Step**
* **规约步骤**：在 **All-Reduce** 操作中，每个节点都会将自己的数据与其他节点的数据进行合并。规约步骤的次数通常与节点数（N）相关。
* **Ring All-Reduce**：需要 $N-1$ 步来完成所有节点的规约过程。
* **Flash All-Reduce**：**通过优化技术（如 Flash Communication），减少为 1 步**。
### **3. Reduce-Scatter**
* **规约-分散步骤**：这一步将数据进行规约（例如求和），然后分散到所有节点。其目的是将每个节点的局部结果合并成全局结果。
* **Ring All-Reduce**：需要 $M / N$ 步来完成规约-分散步骤，M 是数据量，N 是节点数。
* **Flash All-Reduce**： $M (N-1) / N$
### **4. Gather Step**
* **收集步骤**：这一步将各节点的数据聚集在一起，确保每个节点都能获得全局的数据副本。
* **Ring All-Reduce**：收集步骤需要 $N-1$ 步来完成所有节点的数据同步。
* **Flash All-Reduce**：通过技术优化，**将收集步骤减少为 1 步**，进一步减少了通信延迟和开销。
### **5. All-Gather**
* **全收集步骤**：指的是所有节点最终都接收到其他节点的数据，确保每个节点的数据完全一致。
* **Ring All-Reduce**：需要 $M / N$ 步，数据从一个节点到另一个节点传递并收集。
* **Flash All-Reduce**： $M (N-1) / N$。
### **6. QDQ Step**
* **量化-反量化步骤（QD Step）**：在某些情况下，特别是在低位量化中，数据会在发送之前进行量化，接收后进行反量化。
* **Ring All-Reduce**：需要 $N$ 步量化和反量化操作。
* **Flash All-Reduce**：**通过进一步的优化，减少为 2 步，降低了操作复杂度**。
### **总结**：

* **Ring All-Reduce** 是传统的通信方法，每个节点在多个步骤中进行数据交换和同步，通信体积和步骤数较大。

* **Flash All-Reduce** 通过优化和减少步骤（如规约和收集步骤从 $N-1$ 减少到 1，量化-反量化步骤减少到 2）显著提高了效率，尤其适用于大规模并行计算和深度学习推理任务。


#  asymmetric quantization 非对称量化
![[Snipaste_2025-10-27_17-45-13.png]]

# related work
* [[分布式训练中All-Reduce、All-Gather、Reduce-Scatter原理介绍]]
* [[Flash Attention 阅读]]