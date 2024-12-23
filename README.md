# Neural Network from Scratch

## Overview
This project implements a neural network from scratch using only fundamental Python libraries: **NumPy** for numerical computations, **Matplotlib** for visualizations, **Pandas**, and **idx2numpy** for handling and converting image datasets into NumPy arrays. The MNIST dataset is used as the primary dataset for this project, sourced from Kaggle: [MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

The neural network is capable of classifying handwritten digits (0-9) from the MNIST dataset and demonstrates fundamental machine learning principles without relying on high-level libraries such as TensorFlow or PyTorch.

---

## Features
- Built from scratch using NumPy for all mathematical computations.
- Support for multiple layers, including input, hidden, and output layers.
- Configurable number of neurons and activation functions.
- Forward propagation and backpropagation implementation.
- Gradient descent optimization.
- Visualization of training loss and accuracy using Matplotlib.
- Data preprocessing and loading using Pandas and idx2numpy.

---

## Dataset
The MNIST dataset, a collection of 70,000 handwritten digit images (28x28 pixels), is used for training and evaluation. The dataset can be downloaded from Kaggle:

[MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

After downloading the dataset, ensure the files are properly extracted and located in a directory accessible by the script.

---

## Project Structure
```plaintext
├── archive
│   ├── train-images.idx3-ubyte
│   ├── train-labels.idx1-ubyte
│   ├── t10k-images.idx3-ubyte
│   ├── t10k-labels.idx1-ubyte
├── neural_network_Mnist.ipynb  # Jupyter Notebook for the entire implementation
├── README.md                   # Project documentation
```

---

## Installation

### Prerequisites
Ensure Python 3.7 or higher is installed on your system along with the following libraries:

- NumPy
- Matplotlib
- Pandas
- idx2numpy

You can install these dependencies using pip:
```bash
pip install numpy matplotlib pandas idx2numpy
```

---

## Usage

### 1. Data Preparation
Download the dataset from the [MNIST Dataset Kaggle page](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) and place it in the `archive` directory.

### 2. Running the Neural Network
Open the Jupyter Notebook `neural_network_Mnist.ipynb` and run the cells step by step to:
- Load and preprocess the dataset.
- Initialize the neural network with specified hyperparameters.
- Train the model and save training metrics.
- Evaluate the model on the test set.

---

## Visualization
The training process includes the generation of loss and accuracy plots using Matplotlib. These plots help visualize the performance of the neural network over epochs.

---

## Customization
The neural network can be customized by modifying the hyperparameters and architecture in the `neural_network_Mnist.ipynb` notebook. Parameters such as:
- Number of layers
- Number of neurons per layer
- Activation functions (e.g., Sigmoid, ReLU)
- Learning rate
- Batch size

can all be configured to suit your needs.

---

## Limitations
This project is designed as an educational tool and may not perform as well as state-of-the-art neural networks built using advanced libraries like TensorFlow or PyTorch. Optimization techniques are implemented from scratch and may be less efficient for larger datasets or more complex tasks.

---

## Future Work
- Extend the implementation to support convolutional neural networks (CNNs).
- Add support for other datasets and tasks.
- Optimize the backpropagation algorithm for better performance.
- Implement additional optimization algorithms like Adam or RMSprop.

---

## Credits
- MNIST Dataset: [Kaggle - Hojjat K](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- Libraries: NumPy, Matplotlib, Pandas, idx2numpy

---

## License
This project is released under the MIT License.

