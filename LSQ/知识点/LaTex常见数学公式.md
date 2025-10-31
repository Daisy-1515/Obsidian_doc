
# 文章目录

1. 基本语法
2. 常用字符
   3. 希腊字母
   4. 数学运算符
   5. 上下标、上下划线
   6. 字体
   7. 空格、省略号
   8. 括号
9. 常用公式
   10. 特殊符号
   11. 函数符
   12. 矩阵
   13. 方程组
---

## 1. 基本语法

| 语法             | 显示                       |
|------------------|----------------------------|
| 行内公式         | $S = \pi r^2$              |
| 公式             | $\displaystyle S = \pi r^2$ |
| 陈列公式         | $$S = \pi r^2$$            |
| 求圆的面积       | 求圆的面积($S = \pi r^2$)  |
| 优先级           | `${ }$`                    |

**注**：单独显示大括号 `{}` ，需要加反斜线 `\{...\}`

---

## 2. 常用字符

### (1) 希腊字母

| 语法        | 显示    | 语法     | 显示    |
|-------------|---------|----------|---------|
| \alpha      | α       | \beta    | β       |
| \gamma      | γ       | \delta   | δ       |
| \epsilon    | ϵ       | \zeta    | ζ       |
| \eta        | η       | \theta   | θ       |
| \iota       | ι       | \kappa   | κ       |
| \lambda     | λ       | \mu      | μ       |
| \nu         | ν       | \xi      | ξ       |
| \pi         | π       | \rho     | ρ       |
| \sigma      | σ       | \tau     | τ       |
| \upsilon    | υ       | \phi     | ϕ       |
| \chi        | χ       | \psi     | ψ       |
| \omega      | ω       |          |         |

### (2) 数学运算符

| 运算符      | 说明     | 应用举例       | 语法     |
|-------------|----------|----------------|----------|
| +           | 加       | x + y          | $x + y$  |
| -           | 减       | x - y          | $x - y$  |
| \times      | 叉乘     | x × y          | $x \times y$ |
| \cdot       | 点乘     | x ⋅ y          | $x \cdot y$  |
| \ast(*)     | 星乘     | x ∗ y          | $x \ast y$ |
| \div        | 除       | x ÷ y          | $x \div y$ |
| \pm         | 加减     | x ± y          | $x \pm y$  |
| \neq        | 不等于   | x ≠ y          | $x \neq y$ |
| \leq        | 小于等于 | x ≤ y          | $x \leq y$ |
| \geq        | 大于等于 | x ≥ y          | $x \geq y$ |
| \approx     | 约等于   | x ≈ y          | $x \approx y$ |
| \equiv      | 恒等于   | x ≡ y          | $x \equiv y$ |

### (3) 上下标、上下划线

| 语法           | 示例              | 显示          |
|----------------|-------------------|---------------|
| 上标、下标     | $C_n^2$           | $C_n^2$      |
| 矢量           | $\vec a$          | $\vec a$     |
| 字母上^        | $\hat a$          | $\hat a$     |
| 平均数（上划线）| $\overline a$     | $\overline a$|
| 下划线         | $\underline a$    | $\underline a$ |

### (4) 字体

| 字体           | 语法               | 显示        |
|----------------|--------------------|-------------|
| 默认           | \{A\}              | A B C D E F G |
| 等线体         | \sf{A}             | ABCDEFG    |
| 打印机体       | \tt{A}             | ABCDEFG    |
| 罗马体         | \rm{A}             | ABCDEFG    |
| 宋体           | \bf{A}             | ABCDEFG    |
| 黑板粗体       | \Bbb{A}            | ABCDEFG    |
| 意大利体       | \it{A}             | ABCDEFG    |
| 德文字体       | \frak{A}           | ABCDEFG    |

### (5) 空格、省略号

| 空格           | 语法        | 显示        |
|----------------|-------------|-------------|
| 无             | ab          | ab          |
| 小空格         | a\ b        | a b         |
| 4个空格        | a\quad b    | a   b       |

### (6) 括号

| 括号           | 语法           | 显示        |
|----------------|----------------|-------------|
| 小括号         | ( … )         | ( … )      |
| 中括号         | [ … ]         | [ … ]      |
| 大括号         | \{ … \}       | { … }      |
| 尖括号         | \langle … \rangle | ⟨ … ⟩  |
| 绝对值         | \vert … \vert | ∣ … ∣     |

---

## 3. 常用公式

### (1) 特殊符号

| 特殊符号       | 语法        |
|----------------|-------------|
| ∞             | \infty     |
| ∂             | \partial   |
| ∇             | \nabla     |
| △             | \triangle  |
| ∀             | \forall    |
| ∃             | \exists    |
| ¬             | \lnot      |

### (2) 函数符

| 常用运算式     | 语法            | 举例                   |
|----------------|-----------------|------------------------|
| 分式           | \frac{x}{y}     | $\frac{x}{y}$          |
| 根式           | \sqrt[x]{y}     | $\sqrt[x]{y}$          |
| 对数           | \log_n x        | $\log_n x$             |
| 偏导数         | \frac{\partial z}{\partial x} | $\frac{\partial z}{\partial x}$ |
| 极大值         | \max(A, B, C)   | $\max(A, B, C)$        |
| 求和           | \sum_{i=0}^n    | $\sum_{i=0}^n$         |
| 求极限         | \lim_{x \to \infty} | $\lim_{x \to \infty}$ |
| 求积分         | \int_0^\infty f(x)dx | $\int_0^\infty f(x) dx$ |

### (3) 矩阵

| 起始标记           | 结束标记         | 举例                                     |
|--------------------|------------------|------------------------------------------|
| \begin{matrix}     | \end{matrix}     | $$\begin{matrix}1 & 0 & 0\\0 & 1 & 0\\0 & 0 & 1\end{matrix}$$ |
| \begin{pmatrix}    | \end{pmatrix}    | $$\begin{pmatrix}1 & 0 & 0\\0 & 1 & 0\\0 & 0 & 1\end{pmatrix}$$ |

### (4) 方程组
$$
\begin{cases}
a_1x + b_1y + c_1z = d_1 \\
a_2x + b_2y + c_2z = d_2 \\
a_3x + b_3y + c_3z = d_3
\end{cases}
$$
参考链接：
- [CSDN LaTeX公式教程](https://blog.csdn.net/cungudafa/article/details/80301378)
- [CSDN数学公式排版](https://blog.csdn.net/lanxuezaipiao/article/details/44341645)