
# Pretrained Model Fine-Tuning for Plant Classification with Streamlit

This project demonstrates the use of a pretrained deep learning model, such as **EfficientNet**, fine-tuned to classify images into two categories: **cactus** and **flower**. The app is built with Streamlit, providing an interactive interface for model prediction. Leveraging pretrained models allows for accurate and efficient classification by utilizing transfer learning.

## Features

- **Image Classification**: Predicts whether an image is a cactus or a flower.
- **Pretrained Model Utilization**: Leverages transfer learning with EfficientNet or similar models for effective fine-tuning.
- **User-Friendly Interface**: Interactive and easy-to-use interface powered by Streamlit.
- **Environment Management**: Managed with Conda, using Python 3.9.

## Setup and Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MennaElzarqa6/pretrained_models_DL.git
   cd pretrained_models_DL
   ```

2. **Create and Activate Conda Environment**:
   Make sure to have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

   ```bash
   conda create -n plant_classifier python=3.9
   conda activate plant_classifier
   ```

3. **Install Dependencies**:
   Use the provided `requirements.txt` to install all necessary libraries.

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit App**:
   Once all dependencies are installed, launch the app with:

   ```bash
   streamlit run app.py
   ```

   This will open the app in your default web browser, allowing you to upload images for prediction.

## Project Structure

- **app.py**: Main file that contains the Streamlit app code.
- **model/**: Directory for storing the pretrained and fine-tuned model files (e.g., EfficientNet).
- **data/**: Contains subdirectories `cactus` and `flower` for training and validation data.

## Usage

1. Open the app using the instructions in the setup section.
2. Upload an image of either a cactus or a flower.
3. The model will predict and display the category of the uploaded image.

## Model and Data

The model is based on pretrained deep learning architectures like EfficientNet, initially trained on a large image dataset and then fine-tuned on a single-class plant dataset (cactus and flower images). This setup enables high-accuracy predictions without the need for training from scratch.

## Requirements

- Python 3.9
- Streamlit
- TensorFlow or PyTorch (depending on model)
- Other libraries specified in `requirements.txt`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
