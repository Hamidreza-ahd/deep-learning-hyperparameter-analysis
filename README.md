# MNIST Hyperparameter Experiments with MLP

This repository contains a comprehensive set of experiments investigating how basic hyperparameters affect fully‑connected (MLP) neural networks on the MNIST handwritten digit classification task. The notebook systematically compares:

- **Learning rate** (very large → very small)
- **Batch size** (small → large)
- **Model depth and width** (1×128, 2×128, 2×512)

Each experiment includes detailed training logs, four diagnostic plots (training/validation accuracy & loss), and a short analysis. A final comparison table summarises the results and highlights key trends.

---

## 📁 Repository Structure

```
.
├── README.md                 # This file
├── notebook.ipynb            # Main Jupyter notebook
└── requirements.txt          # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Matplotlib
- Jupyter (to run the notebook)

Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Notebook
1. Clone this repository.
2. Launch Jupyter:

   ```bash
   jupyter notebook notebook.ipynb
   ```
3. Run all cells to reproduce the experiments and plots.

> **Note:** The MNIST dataset is downloaded automatically via Keras the first time you run the notebook.

---

## 🧪 Experiments Overview

All models are trained for 20 epochs using the **SGD** optimizer (unless stated otherwise). The input images are flattened to 784‑dimensional vectors and scaled to `[0,1]`. 10% of the training data is used for validation.

| Model | Architecture     | Optimizer / LR   | Batch Size | Key Observation |
|-------|------------------|------------------|------------|------------------|
| 1     | 1×128            | SGD (default)    | 128        | Baseline – steady convergence |
| 2     | 1×128            | SGD (lr=1.0)     | 128        | Very fast, near‑perfect training, slight overfitting |
| 3     | 1×128            | SGD (lr=1e-6)    | 128        | No learning – accuracy near random |
| 4     | 1×128            | SGD (default)    | 16         | Noisy updates, good generalization |
| 5     | 1×128            | SGD (default)    | 128        | Repeat of baseline – consistent |
| 6     | 1×128            | SGD (default)    | 512        | Smoother gradients, slower convergence, lower final accuracy |
| 7     | 2×128            | SGD (default)    | 128        | Deeper network – improved performance |
| 8     | 2×512            | SGD (default)    | 128        | Widest & deepest – best accuracy, diminishing returns |

---

## 📊 Final Results Table

| Model | Architecture | Optimizer / LR | Batch | train_acc | val_acc | train_loss | val_loss |
|-------|--------------|----------------|-------|-----------|---------|------------|----------|
| 1     | 1×128        | SGD (default)  | 128   | 0.9340    | 0.9495  | 0.2341     | 0.1923   |
| 2     | 1×128        | SGD (1.0)      | 128   | 0.9998    | 0.9800  | 0.0034     | 0.0975   |
| 3     | 1×128        | SGD (1e-6)     | 128   | 0.0834    | 0.0785  | 2.4553     | 2.4566   |
| 4     | 1×128        | SGD (default)  | 16    | 0.9814    | 0.9777  | 0.0685     | 0.0810   |
| 5     | 1×128        | SGD (default)  | 128   | 0.9346    | 0.9490  | 0.2359     | 0.1936   |
| 6     | 1×128        | SGD (default)  | 512   | 0.9000    | 0.9208  | 0.3663     | 0.2987   |
| 7     | 2×128        | SGD (default)  | 128   | 0.9477    | 0.9588  | 0.1854     | 0.1548   |
| 8     | 2×512        | SGD (default)  | 128   | 0.9537    | 0.9653  | 0.1615     | 0.1358   |

---

## 🔍 Key Takeaways

- **Learning rate** must be balanced: too high (1.0) leads to rapid overfitting; too low (1e-6) stalls learning completely.
- **Batch size** affects convergence and generalization:
  - Small batches (16) introduce gradient noise that acts as implicit regularisation, yielding good validation accuracy.
  - Large batches (512) produce smoother gradients but converge to lower final accuracy within the same epoch budget.
- **Network capacity** matters:
  - Deeper (2×128) and wider (2×512) models outperform the shallow baseline.
  - Gains from further capacity increase (2×512 vs 2×128) are modest, showing diminishing returns.
- The **best overall model** (2×512) achieves ~96.5% validation accuracy, confirming that a moderately deep and wide MLP is well suited for MNIST.

---

## 🙌 Acknowledgements

- The [MNIST database](http://yann.lecun.com/exdb/mnist/) of handwritten digits.
- TensorFlow / Keras for providing an easy‑to‑use deep learning framework.

Feel free to fork, experiment further, and open issues or pull requests!
