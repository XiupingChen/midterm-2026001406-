# 期中考试

## 1. 说明

下面是 Part A 编程部分，包括四道题目：

- **A1. 欧拉法与改进欧拉法**：实现常微分方程初值问题的两种数值求解方法。
- **A2. RandomNormalVector**：实现正态随机向量类、归并排序、样本标准差和输出运算符。
- **A3. 图像块平均降采样**：利用 RcppArmadillo 实现灰度图像的分块平均池化。
- **A4. 加权最小二乘**：利用 RcppArmadillo 实现加权回归，并与 R 的参考结果进行比较。

A1、A2 使用 C++11 实现，A3、A4 使用 C++ 与 RcppArmadillo 实现。四道题均已完成编译和运行测试。

## 2. 文件结构

```text
starter-code/
└── part-a/
    ├── ode-solvers.cpp
    ├── random-normal-vector.cpp
    ├── block-pooling.cpp
    ├── block-pooling.R
    ├── weighted-least-squares.cpp
    └── weighted-least-squares.R
```

其中，`.cpp` 文件包含算法实现，`.R` 文件用于编译调用、数值验证和绘图。

## 3. 实现方法

### A1. 欧拉法与改进欧拉法

考虑初值问题：

$$
\frac{dy}{dt}=y-t^2+1,\qquad y(0)=0.5,\quad t\in[0,2].
$$

解析解为：

$$
y(t)=(t+1)^2-\frac12e^t.
$$

在 `ode-solvers.cpp` 中实现 `euler()` 和 `improvedEuler()`。

普通欧拉法采用迭代公式：

$$
y_{i+1}=y_i+hf(t_i,y_i).
$$

改进欧拉法首先计算预测值：

$$
y_{i+1}^{p}=y_i+hf(t_i,y_i),
$$

随后使用起点与预测终点的平均斜率进行校正：

$$
y_{i+1}=y_i+\frac h2
\left[f(t_i,y_i)+f(t_{i+1},y_{i+1}^{p})\right].
$$

分别采用步长 $h=0.1$ 和 $h=0.05$，得到长度为 21 和 41 的数值解序列。使用题目提供的离散参考结果进行逐点验证，并计算相对于解析解的最大绝对误差。

### A2. RandomNormalVector

在 `random-normal-vector.cpp` 中完成 `RandomNormalVector` 类的相关功能。

主要实现包括：

1. 带参数构造函数 `RandomNormalVector(n, mean, stddev)`，生成指定均值和标准差的正态随机向量。
2. `mergeSort()` 使用分治与递归实现归并排序，不调用标准库排序函数替代算法。
3. `sd()` 计算样本标准差，使用 $n-1$ 作为方差分母。
4. 重载 `operator<<`，按照当前存储顺序输出向量中的所有元素。

对于指定参数 $\mu$ 和 $\sigma$，正态随机数通过以下变换生成：

$$
X_i=\mu+\sigma Z_i,\qquad Z_i\sim N(0,1).
$$

测试包括构造函数、排序、均值、样本标准差、输出运算符及正态随机向量示例。

### A3. 图像块平均降采样

在 `block-pooling.cpp` 中实现：

```cpp
arma::mat blockPool(const arma::mat& img, int blockSize);
```

将输入灰度图像矩阵划分为不重叠的 $b\times b$ 图像块，计算每个块内像素的平均值，并生成降采样后的矩阵。

输出矩阵尺寸为：

$$
\left\lceil\frac{m}{b}\right\rceil
\times
\left\lceil\frac{n}{b}\right\rceil.
$$

使用嵌套循环遍历输出矩阵与对应的输入区域。对于边缘不足完整块大小的区域，根据实际包含的像素数量计算平均值，不进行补零。

同时处理以下情况：

- 当 `blockSize = 1` 时返回原始图像。
- 当块大小不小于图像的行数和列数时，返回全图平均值。
- 检查输入矩阵与块大小的有效性。

使用 `block-pooling.R` 对小矩阵和边缘区域进行测试，并绘制原图以及块大小为 2、4、8 时的降采样对比图。

### A4. 加权最小二乘

在 `weighted-least-squares.cpp` 中实现：

```cpp
Rcpp::List weightedLeastSquares(
    const arma::mat& X,
    const arma::vec& y,
    const arma::vec& weights
);
```

求解加权最小二乘问题：

$$\hat{\beta}=\arg\min_{\beta}\sum_{i=1}^{n}w_i(y_i-x_i^\top\beta)^2$$

采用行缩放方式完成加权：

$$
X_{\mathrm{w}}=W^{1/2}X,
\qquad
y_{\mathrm{w}}=W^{1/2}y.
$$

其中，权重矩阵为：

$$
W=\mathrm{diag}(w_1,\ldots,w_n).
$$

实现时不显式构造 $n\times n$ 的权重矩阵，而是将设计矩阵的第 $i$ 行和响应变量的第 $i$ 个元素同时乘以 $\sqrt{w_i}$。

使用 Armadillo 的 `solve()` 求解变换后的最小二乘问题，不显式计算矩阵逆，也不额外添加截距或对数据进行标准化。

函数返回：

- `coefficients`：回归系数；
- `fitted`：拟合值；
- `weighted_rss`：加权残差平方和。

使用 `weighted-least-squares.R` 与 R 的 `lm()` 进行对比，并验证全 1 权重和权重整体缩放两种情况。

## 4. 编译与运行

### 4.1 运行环境

本次程序使用以下环境进行测试：

- macOS：用于 A1、A2 的 C++ 编译与运行。
- C++ 编译器：`g++`，使用 C++11 标准。
- Docker Desktop：运行 Rocker RStudio Server 容器。
- R：4.6.1。
- Rcpp、RcppArmadillo：用于 A3、A4 的 C++ 编译和 R 调用。

所有命令均在 `starter-code/part-a/` 目录下执行。

### 4.2 A1 编译与运行

```bash
g++ -std=c++11 ode-solvers.cpp -o ode-solvers
./ode-solvers
```

### 4.3 A2 编译与运行

```bash
g++ -std=c++11 random-normal-vector.cpp -o random-normal-vector
./random-normal-vector
```

### 4.4 A3 编译与运行

在 RStudio 中设置工作目录为 `starter-code/part-a/`，运行：

```r
source("block-pooling.R")
```

测试脚本会通过 `Rcpp::sourceCpp()` 编译对应的 C++ 文件，然后运行数值测试并生成对比图。

### 4.5 A4 编译与运行

同样在 RStudio 中运行：

```r
source("weighted-least-squares.R")
```

测试脚本完成 C++ 编译、加权回归计算以及与 R 参考结果的比较。

## 5. 测试结果

### A1. 欧拉法与改进欧拉法

两种数值方法均通过参考离散解的逐点验证，最大绝对误差如下：

| 方法 | $h=0.1$ | $h=0.05$ |
|---|---:|---:|
| Euler | 0.241972 | 0.127466 |
| Improved Euler | 0.0189048 | 0.00481986 |

在相同步长下，改进欧拉法的最大绝对误差明显小于普通欧拉法。当步长从 0.1 减小至 0.05 时，两种方法的误差均下降，其中改进欧拉法的误差下降幅度更大。

### A2. RandomNormalVector

运行结果：

```text
PASS: parameterized constructor
PASS: merge sort
PASS: mean
PASS: sample standard deviation
PASS: stream output
PASS: Normal examples
```

各项测试均通过，包括参数化构造函数、归并排序、均值与样本标准差计算以及向量输出。

### A3. 图像块平均降采样

R 测试脚本输出：

```text
Small matrix, partial edge, identity and whole-image checks passed.
```

测试验证了小矩阵池化、边缘不完整块、`blockSize = 1` 以及整幅图像平均值的情况。

同时生成原始图像及不同块大小的降采样结果。随着块大小增加，图像分辨率降低，图像边缘逐渐表现出更明显的块状特征。

### A4. 加权最小二乘

测试得到的回归系数为：

```text
0.9076389  0.9729167
```

加权残差平方和：

```text
0.6704861
```

五个拟合值为：

```text
0.9076389
1.8805556
2.8534722
3.8263889
4.7993056
```

测试脚本输出：

```text
R reference, all-one weights and rescaled-weight checks passed.
```

C++ 计算结果与 R 的参考结果一致，回归系数、拟合值和加权残差平方和均通过数值验证，全 1 权重及权重整体缩放测试也均通过。

## 6. 完成情况

Part A 的四道题均已完成实现，并成功通过题目提供的现有测试。

A1、A2 在 macOS 终端通过 C++11 编译和运行；A3、A4 在 Docker 的 RStudio Server 环境中使用 RcppArmadillo 编译执行。

所有测试结果以实际编译和运行输出为依据。

