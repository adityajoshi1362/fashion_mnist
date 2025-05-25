# 🧥 Fashion MNIST Classification with CNN
This project showcases a Convolutional Neural Network (CNN) built using TensorFlow and Keras to classify grayscale images from the Fashion MNIST dataset. The dataset consists of 70,000 images across 10 categories of clothing items.

🎯 Accuracy Achieved
✅ Test Accuracy: ~92%

✅ Validation Accuracy: Up to 91.8% during training

🧠 Model Architecture
A deep CNN was designed using keras.Sequential, composed of multiple convolutional layers, max pooling layers, and fully connected layers with dropout regularization.


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
📆 Epochs: 10

📦 Batch Size: 64

📉 Validation Split: 10%

The dataset was normalized and reshaped to match the input requirements of the CNN. Training accuracy improved rapidly, surpassing 90% within the first few epochs.

📦 Dataset Details
The Fashion MNIST dataset contains:

👕 Training Images: 60,000

👟 Testing Images: 10,000

🖼 Image Size: 28x28 pixels (grayscale)

🏷️ Number of Classes: 10

👚 Class Labels:
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

🖼 Sample Visualizations
The project includes code to visualize both:

✅ Training samples with their labels

✅ Model predictions on test images with true vs predicted class names

📌 Requirements
Python 3.x

TensorFlow ≥ 2.x

NumPy

Matplotlib

Install via:


    pip install tensorflow numpy matplotlib
🚀 Future Improvements
🔁 Implement early stopping and learning rate decay

🧪 Add a confusion matrix and classification report

🧠 Experiment with data augmentation

📦 Export the model using TensorFlow Lite or ONNX for deployment

👤 Author
Aditya Joshi
📬 Feel free to connect for suggestions, feedback, or collaboration opportunities!
