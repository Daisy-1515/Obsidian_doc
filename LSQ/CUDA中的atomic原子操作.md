## 原子操作简介

原子操作是指在执行过程中不会被任何其他任务或事件中断的最小执行单位。在CUDA中，原子操作用于确保对共享变量的“读取-修改-写入”操作是一个不可中断的最小事务。该机制保证了在并行线程之间对共享变量的读写操作的互斥性，从而确保每次对变量的操作结果的正确性。

**原子操作的特点：**
* 确保在并行线程中，只有一个线程能够对共享变量进行操作。
* 当一个线程操作某个变量时，其他线程只能等待前一个线程完成操作后才能继续。
* 原子操作提供了数据的一致性和安全性，但代价是性能上的牺牲。
---
## 常用的原子操作
1. **加法操作——`atomicAdd()`**
* 对位于全局或共享存储器的32位或64位整数执行加法操作，并将结果存储回原地址。
* 返回值为加法操作前的旧值。
```cpp

int atomicAdd(int* address, int val);

unsigned int atomicAdd(unsigned int* address, unsigned int val);

unsigned long long int atomicAdd(unsigned long long int* address, unsigned long long int val);

```
2. **减法操作——`atomicSub()`**
* 对位于全局或共享存储器的32位整数执行减法操作，并将结果存储回原地址。
* 返回值为减法操作前的旧值。
```cpp

int atomicSub(int* address, int val);

unsigned int atomicSub(unsigned int* address, unsigned int val);

```
3. **交换操作——`atomicExch()`**
* 将一个值存储到指定地址，并返回存储前的值。
```cpp

int atomicExch(int* address, int val);

unsigned int atomicExch(unsigned int* address, unsigned int val);

unsigned long long int atomicExch(unsigned long long int* address, unsigned long long int val);

float atomicExch(float* address, float val);

```
4. **最小值操作——`atomicMin()`**
* 计算当前值和指定值的最小值，并将其存储回指定地址。
```cpp

int atomicMin(int* address, int val);

unsigned int atomicMin(unsigned int* address, unsigned int val);

```
5. **最大值操作——`atomicMax()`**
* 计算当前值和指定值的最大值，并将其存储回指定地址。
```cpp

int atomicMax(int* address, int val);

unsigned int atomicMax(unsigned int* address, unsigned int val);

```
6. **增量操作——`atomicInc()`**
* 将存储的值加1，如果达到最大值则重置为0。
```cpp

unsigned int atomicInc(unsigned int* address, unsigned int val);

```
7. **减量操作——`atomicDec()`**
* 将存储的值减1，如果达到最小值则重置为指定的值。
```cpp

unsigned int atomicDec(unsigned int* address, unsigned int val);

```
8. **比较并交换——`atomicCAS()`**
* 比较指定地址的值是否等于指定的值，如果相等，则将新的值写入地址，并返回旧值。
```cpp

int atomicCAS(int* address, int compare, int val);

unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val);

unsigned long long int atomicCAS(unsigned long long int* address, unsigned long long int compare, unsigned long long int val);

```
9. **与操作——`atomicAnd()`**
* 执行与操作，将计算结果存储回指定地址。
```cpp

int atomicAnd(int* address, int val);

unsigned int atomicAnd(unsigned int* address, unsigned int val);

```
10. **或操作——`atomicOr()`**
* 执行或操作，将计算结果存储回指定地址。
```cpp

int atomicOr(int* address, int val);

unsigned int atomicOr(unsigned int* address, unsigned int val);

```
11. **异或操作——`atomicXor()`**
* 执行异或操作，将计算结果存储回指定地址。
```cpp

int atomicXor(int* address, int val);

unsigned int atomicXor(unsigned int* address, unsigned int val);

```
---
### 代码示例：使用 `atomicAdd()` 进行加法操作

```cpp

#include <stdio.h>

#include <stdlib.h>

#include <cuda_runtime.h>

__global__ void histo_kernel(unsigned int *histo)

{

int atomic_value = atomicAdd(histo, 1);

printf("atomic_value: %d, histo: %d\n", atomic_value, *histo);

}

int main(void)

{

int threadSum = 3;

// 分配内存并拷贝初始数据

unsigned int *dev_histo;

cudaMalloc((void**)&dev_histo, sizeof(int));

cudaMemcpy(dev_histo, &threadSum, sizeof(int), cudaMemcpyHostToDevice);

histo_kernel<<<1, 1>>>(dev_histo);

// 数据拷贝回CPU内存

cudaMemcpy(&threadSum, dev_histo, sizeof(int), cudaMemcpyDeviceToHost);

cudaFree(dev_histo);

return 0;

}

```

**输出：**

```

atomic_value: 3, histo: 4

```

---

通过使用CUDA中的原子操作，我们可以确保在多线程并行计算中对共享内存的操作是安全且不被中断的，避免数据竞争问题，但这也可能带来性能上的代价。