# Email Syntax Validation Neural Network

A high-performance, character-level **Bidirectional Recurrent Neural Network (RNN)**, specifically using **Long Short-Term Memory (LSTM)** units, combined with a character-level **Embedding** front-end. built natively in TensorFlow to classify email address structures as valid or invalid. The system achieves **99.xx% validation accuracy** by moving past simple structural shortcuts to learn robust sequential text dependencies.

---

## 📁 Project Architecture

The project is structured as a fully decoupled, portable Python package with a centralized control dashboard:

```text
my_email_project/
├── data/
│   └── emails.csv          # Balanced dataset (10k valid, 10k invalid edge cases)
├── src/
│   └── neural_network/
│       ├── __init__.py     # Package initialization marker
│       ├── __main__.py     # Parameterized entry point (sys.argv execution router)
│       ├── network.py      # Netwrok class to wrapp tensor flow model
│       └── vocab/
│           ├── map.json    # 85-token core vocabulary configuration array
│           └── vocab.py    # Vocab class to handle the allowed characters map
├── config.py               # Central control dashboard (Hyperparameters & paths)
├── pyproject.toml          # Modern setuptools build system declaration file
└── README.md
```

---

## 🚀 Setup and Installation

Follow these steps to initialize an isolated environment and install the package requirements:

### 1. Clone and Navigate
```bash
git clone https://github.com/appostolov/neural-network-python-tensor-flow.git
cd neural-network-python-tensor-flow
```

### 2. Install in Editable Development Mode
Execute the modern build installation directly on the root directory. This downloads `tensorflow`, `numpy` and `pandas` and hooks up a live shortcut to your source files:
```bash
pip install -e .
```

---

## 🕹️ Package Execution Interface

The package handles execution routing dynamically based on text parameters passed directly from the terminal line.

### ➔ Train the Model (80/20 Validation Split)
The following command creates the model `Network.create()`, prepares the dataset `Network.dataset()`, trains the model `Network.train()` and saves the calculated weights `Network.train()` in `models/model.keras`:
```bash
python -m src.neural_network train
```

### ➔ Test Specific Strings (Real-Time Inference)
The following command opens the model `Network.open()` from `models/model.keras`, runs it `Network.run()` and prints the output prediction in the console:
```bash
python -m src.neural_network run robert.global.testing123@yahoo.com
```

---

## 📊 Core Optimizations and Safeguards

* **Native Serialization Engine:** Employs the modern, recommended `.keras` saving framework rather than legacy directory layouts or loose `.h5` files. This locks down your architecture configurations, tokenization mapping thresholds, and weight statistics into a single portable binary file.
* **Bidirectional Contextual Feature Extraction:** Utilizes a Recurrent Neural Network (RNN) core featuring Bidirectional LSTMs. By tracking data streams forward (left-to-right) and backward (right-to-left) simultaneously, the network evaluates local characters against global indicators—such as verifying symbol balance around domain names and handling sequential text dependencies natively.
* **Overfitting Regularization Head:** Combines a specialized hidden Dense layer with an active Dropout layer (`layers.Dropout`) right before the final binary classification decision. This introduces a regularized bottleneck that penalizes the memorization of specific training strings and forces the model to generalize across global structural boundaries.
