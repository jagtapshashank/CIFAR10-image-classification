# Introduction
This repository contains the code and resources for a deep learning model designed for 10-class image classification on the CIFAR-10 dataset, developed as part of Deep Learning class at Texas A&M University. The model, MyNet, incorporates a novel "ImprovedBlock" architecture that enhances feature extraction and classification accuracy, achieving robust performance on challenging image classification tasks.

# Model Development

The model utilizes a custom block structure, **ImprovedBlock**, which is specifically designed to enhance deep network training. This block includes residual connections that mitigate the vanishing gradient problem, facilitating the training of deeper networks. Each block also integrates batch normalization and ReLU activation to stabilize the training process and accelerate convergence. The architecture is composed of three main layers of stacked ImprovedBlock modules, progressively extracting more complex features. After these layers, global average pooling is applied to reduce spatial dimensions before a fully connected layer handles the final classification.

Training was conducted on the CIFAR-10 dataset using **Stochastic Gradient Descent (SGD)** with momentum, leveraging a batch size of 128 for stable gradient flow and efficient GPU utilization. The learning rate schedule starts at 0.1, decreasing at epochs 75, 125, and 200 to encourage smooth convergence. Data augmentation techniques, such as random horizontal flip, random crop, and random erasing, were implemented to improve generalization by diversifying the training data. These strategies collectively contribute to the model’s robust performance on the image classification task.

# Performance Metrics

After training for 200 epochs, the model achieved the following:

Training Loss: 0.0822<br>
Validation Loss: 0.4220<br>
Validation Accuracy: 91.46%<br>
Test Accuracy on CIFAR-10: 90.13%

The final model is saved based on the checkpoint with the highest validation accuracy.

# How to run the code

Training:<br>
Write model/train configs in Configs.py<br>
Use: `main.py train <train data dir>`<br>
Instruction I used: `!python3 main.py train cifar-10-batches-py`<br>

Testing:<br>
Write the model name generated in train in Config.py<br>
Use: `main.py test <train data dir>`<br>
Instruction I used: `!python3 main.py test cifar-10-batches-py` <br>

Predicting:<br>
Write the model name generated in train in Config.py<br>
Use: `main.py predict <private test data dir> --results_dir <results filename>`<br>
Instruction I used: `!python3 main.py predict private_test_images_2024.npy --result_dir results/predictions.npy`<br>

# Conclusion

My custom model demonstrated strong generalization on CIFAR-10, showcasing the effectiveness of the ImprovedBlock architecture and data augmentation techniques in enhancing model robustness. This project contributes to deep learning methodologies for image classification, focusing on innovative architecture design and training strategies.

## Caution: This project demands significant computational power for model training and should only be run on systems equipped with a powerful GPU.
