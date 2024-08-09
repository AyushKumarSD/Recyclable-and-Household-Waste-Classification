# Recyclable and Household Waste Classification

## Project Overview
The "Recyclable and Household Waste Classification" project, developed under the guidance of Prof. Rakesh Das at the University of San Diego for the course AAI – 501: Introduction to Artificial Intelligence, aims to leverage state-of-the-art computer vision and machine learning techniques to enhance waste management and recycling processes. By automating waste classification, this project addresses critical environmental sustainability challenges and reduces reliance on manual sorting processes which are labor-intensive and error-prone.

## Authors
- Ayush Kumar
- Vedant Kumar
- Satyam Kumar

## Abstract
Traditional waste management relies heavily on manual labor which is not only inefficient but also prone to errors. This project utilizes Convolutional Neural Networks (CNN) and pre-trained models like VGG16, ResNet50V2, and InceptionV3 to automate the classification of waste. These models have shown remarkable accuracy in classifying waste into 30 distinct categories, significantly outperforming traditional methods and thereby promoting more effective recycling and sustainable practices.

## Features

### Data Processing
- **Real-Time Data Handling:** Capable of processing live data feeds to classify waste in real time.
- **Comprehensive Data Preprocessing:** Includes normalization, resizing, and augmentation techniques to prepare data for training.

### Augmentation
- **Advanced Augmentation Techniques:** Implements rotations, flips, and cropping to enrich the training dataset, enhancing the robustness of the models.

### Model Training
- **Utilization of Advanced Models:** Employs deep learning models and transfer learning techniques to achieve high accuracy in waste classification.
- **Performance Evaluation:** Rigorous testing with validation sets to tune models and assess their performance accurately.

### Deployment
- **Scalable Solution:** Models are deployed to handle large-scale data effectively, suitable for integration into existing waste management systems.
- **Interactive Dashboard:** An interactive dashboard for real-time monitoring and management of waste classification results.

## Installation

Clone the project repository:
```bash
git clone https://github.com/yourusername/Recyclable-Waste-Classification.git
cd Recyclable-Waste-Classification
```
## Set up a virtual environment and install dependencies:

```
python -m venv venv
source venv/bin/activate  # Use `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

## Usage
## Run the application:

``` 
streamlit run python waste_classifier_app.py
```


