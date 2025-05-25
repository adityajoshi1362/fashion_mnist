# 🧥 Fashion MNIST Classification with CNN

This project demonstrates a convolutional neural network (CNN) built using TensorFlow and Keras to classify images from the Fashion MNIST dataset. The dataset contains grayscale images of 10 different categories of clothing items.

📊 Accuracy Achieved
✅ Test Accuracy: ~92%
✅ Validation Accuracy: Up to 91.8% during training


🧠 Model Architecture
The model is a deep CNN built using keras.Sequential. Below is the architecture:

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])



🧪 Training Summary
Epochs: 10

Batch Size: 64

Validation Split: 10%

The model was trained on the normalized and reshaped Fashion MNIST data. During training, the model quickly improved to over 90% accuracy within a few epochs.


📦 Dataset

It contains 70,000 images of clothing items:

Training images: 60,000

Testing images: 10,000

Image size: 28x28 pixels (grayscale)

Classes: 10 categories:

T-shirt/top

Trouser

Pullover

Dress

Coat

Sandal

Shirt

Sneaker

Bag

Ankle boot


🖼 Sample Visualization
The project includes code to visualize training samples and model predictions.

📌 Requirements
Python 3.x

TensorFlow >= 2.x

NumPy

Matplotlib

📬 Future Improvements
Implement early stopping and learning rate scheduling

Experiment with data augmentation

Add confusion matrix and classification report

Export the model for deployment (e.g., TensorFlow Lite or ONNX)

👤 Author
Aditya Joshi
Feel free to reach out for suggestions, feedback, or collaborations!

