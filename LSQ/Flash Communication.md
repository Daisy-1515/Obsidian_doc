![[Pasted image 20251027144345.png]]
exp：


1. **RMSNorm**：这是一个归一化操作，应用于输入张量（QKV——查询、键、值）。它为模型中的后续层做准备。
2. **QKV和SelfAttention**：这部分表示用于自注意力机制的查询、键和值张量。SelfAttention模块执行注意力计算，这是像LLaMA这类基于Transformer的模型的核心。
3. **Flash All-Reduce**：这是图中展示的核心创新。Flash All-Reduce是一种高效的分布式计算通信技术，旨在加速张量并行。图中强调了在前向传播过程中使用这一技术，其中注意力机制（QKV）的结果被聚合并在设备间通信。使用低位量化（例如INT4、INT6）有助于减少传输的数据量，从而提高速度，同时不牺牲太多准确性。
4. **Gate with Swish**：这个模块可能指的是一种门控机制，可能与激活函数Swish相关，Swish用于在模型中引入非线性。
5. **Down和Up Projections**：这些指的是在Transformer模型的前馈网络中应用于张量值的转换操作。
总之，这个图表概述了张量如何通过不同的操作，如归一化、自注意力和投影，进行处理，重点介绍了如何通过Flash All-Reduce优化通信，这减少了延迟并提高了分布式模型推理中的效率。这对于大规模模型至关重要，因为通信瓶颈常常成为限制因素。

# related work
[[分布式训练中All-Reduce、All-Gather、Reduce-Scatter原理介绍]]
[[Flash Attention 阅读]]