This repository contains the official implementation for the paper:

**Machine Unlearning in the Era of Quantum Machine Learning: An Empirical Study**

We present the first comprehensive empirical study of machine unlearning (MU) in hybrid quantum-classical neural networks. While MU has been extensively explored in classical deep learning, its behavior within variational quantum circuits (VQCs) and quantum-augmented architectures remains largely unexplored.

In this work, we:
- Adapt a broad suite of machine unlearning methods to quantum settings, including:
  - gradient-based methods
  - distillation-based methods
  - regularization-based methods
  - certified unlearning techniques
- Propose two novel unlearning strategies specifically tailored for hybrid quantum–classical models.
- Evaluate unlearning under both **subset removal** and **full-class deletion** scenarios.

Experiments are conducted on **Iris**, **MNIST**, and **Fashion-MNIST** datasets using hybrid quantum-classical neural networks. Results show that quantum models can support effective unlearning, but outcomes depend strongly on:
- circuit depth
- entanglement structure
- task complexity

Shallow VQCs exhibit high intrinsic stability with limited memorization, while deeper hybrid models reveal stronger trade-offs between utility, forgetting strength, and alignment with oracle retraining. Across settings, methods such as **EU-k**, **LCA**, and **Certified Unlearning** provide the most consistent balance across metrics.

This repository is intended to serve as a **baseline and benchmark** for future research in quantum machine unlearning.
