
# Flat Tire Detection

Flat Tire Detection is an ML image classifier that uses pictures of tires to identify flat tires. Designed for household implementation, it automatically notifies the owner when a flat tire is detected.

## Features
- **Image Classification**: Detects whether a tire is flat, full, or missing.
- **Automated Notifications**: Notifies users upon detection of a flat tire.
- **Parallel Processing**: Efficient image loading using `pymp` for multiprocessing.
- **Deep Learning Model**: Uses CNN architecture with TensorFlow/Keras.
- **Visualization**: Training and validation accuracy/loss graphs.

## Tech Stack
- **Machine Learning**: TensorFlow, Keras
- **Preprocessing**: OpenCV, NumPy
- **Data Handling**: Scikit-learn
- **Parallel Processing**: pymp
- **Visualization**: Matplotlib

## Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- Scikit-learn
- pymp

### Clone the Repository
```sh
git clone https://github.com/meera0709/flat-tire-detection.git
cd flat-tire-detection
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

### Run the Model
1. Ensure dataset is structured as:
```
cao_dataset/
    flat.class/    # Images of flat tires
    full.class/    # Images of full tires
    no-tire.class/ # Images without tires
```
2. Run the main script:
```sh
python main.py
```
3. Enter an image path for classification when prompted.

### Expected Output
- The trained model prints validation accuracy and loss.
- Prediction for user-input images is displayed.

## Usage
1. **Train the Model**: The script loads and preprocesses images before training a CNN.
2. **Evaluate Performance**: Accuracy and loss graphs are generated.
3. **Classify New Images**: Enter a tire image path for classification.

## Deployment
For real-world applications, consider deploying using:
- Flask/FastAPI for an API endpoint.
- AWS Lambda or Google Cloud for serverless inference.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m 'Add feature X'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request.

## License
MIT License. See `LICENSE` for details.

## Contact
For questions or support, contact: [your-email@example.com]
```
