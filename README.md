# Mean-Centered Skewness Index (MCSI)

The **Mean-Centered Skewness Index (MCSI)** is a statistical operator for analyzing the **skewness** of tensor data. Built with PyTorch and C++, the `mcsi` function efficiently quantifies the directional imbalance of tensor values relative to their mean across one or more dimensions.

---

## Key Features
- Compatible with PyTorch tensors.
- Configurable reduction dimensions (`dim`) and shape preservation (`keepdim`).
- Efficient implementation as a PyTorch extension using C++ and Pybind11.
- Provides meaningful insights into the asymmetry of data distributions.

---

## Installation
To use the `mcsi` module, you can install it directly via pip:

```pip install mcsi```

---

## Usage
# Importing the Module
To use the `mcsi` function in your python code, import the module as follows:
```python
import mcsi
```

## Mathematical Representation

Let $\mathbf{X} \in \mathbb{R}^{n_1 \times n_2 \times \dots \times n_k}$ represent an input tensor. The **Mean-Centered Skewness Index** for dimension(s) $d$ is defined as:

$$
\text{MCSI}(\mathbf{X}, d) = \frac{N_-(\mathbf{X}, d) - N_+(\mathbf{X}, d)}{N(\mathbf{X}, d)}
$$

Where:
- $N_-(\mathbf{X}, d)$: Number of elements in $\mathbf{X}$ less than the mean along the specified dimensions $d$.
- $N_+(\mathbf{X}, d)$: Number of elements in $\mathbf{X}$ greater than the mean along $d$.
- $N(\mathbf{X}, d)$: Total number of elements in $\mathbf{X}$ along $d$.

The **mean** along dimension(s) $d$ is given by:

$$
\mu(\mathbf{X}, d) = \frac{1}{N(\mathbf{X}, d)} \sum_{i \in d} X_i
$$

MCSI effectively measures the imbalance in the count of values above and below the mean, normalized by the total number of elements.

---

## Function Signature

```python
mcsi(input: torch.Tensor, dim: Optional[Union[int, List[int]]] = None, keepdim: bool = False) -> torch.Tensor
