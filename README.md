# Interactive Chatbot Application

## Overview

The main objective of this project was to create a functional and interactive chatbot capable of generating coherent
and contextually appropriate responses. This involves both the technical aspects of machine learning model training
and the practical aspects of GUI application development.

## Project Components

1. **Data processing** - Exploratory Data Analysis (EDA), data preprocessing and model development in Jupyter
Lab notebook:  
- `development.ipynb`.
2. **Chatbot application development** and testing in a Python-based notebook:  
- `chatbot_app.ipynb`.
3. **Conversion to a standalone Python script** for a fully working GUI chatbot application:
- `chatbot_app.py`.
4. **Sample files** used for initial stages of preprocessing, training of different models, and in different
environments, located in the `samples/` folder.

## Key Features

- Uses a sequence-to-sequence bidirectional encoder-decoder Keras model to generate conversations.
- Responses are generated with two different functions: argmax and beam search.
- Runs on a customtkinter-based GUI with a conversation display field, input field, and two fields of samples
for easy input testing.

## Requirements

- **Python Version**: 3.10.14
- **TensorFlow Version**: 2.10.0
- **TensorFlow Keras Version**: 2.10.0

The application was tested and runs on several newer environments as well.

## Installation Instructions

Follow these steps to set up the application locally:

1. **Ensure you have Python 3.10 installed** on your system. You can download it from [Python's official website](https://www.python.org/downloads/).
2. **Download the project** from its [GitHub repository](https://github.com/Tomas4python/chatbot) and extract it into a new folder or use GIT to clone the project:
   - Open **Command Prompt** or **PowerShell**.
   - Navigate to the project folder:
     ```
     cd path\to\project\folder
     ```
   - Clone the project:
     ```
     git clone "https://github.com/Tomas4python/chatbot.git"
     ```

3. **Set Up a Virtual Environment**:
   - Navigate to the project folder:
     ```
     cd path\to\project\folder
     ```
   - Create a new virtual environment named `venv`:
     ```
     python -m venv venv
     ```
   - Activate the virtual environment:
     ```
     .\venv\Scripts\activate
     ```

4. **Install Dependencies**:
   - Navigate to the project folder:
     ```
     cd path\to\project\folder
     ```
   - Install all required dependencies:
     ```
     pip install -r requirements.txt
     ```

By following these steps, you'll have a clean and isolated Python environment for running and testing this project.

## Launching the Application

Navigate to the project folder and activate the virtual environment. Run the application by executing:
```
python chatbot_app.py
```

## Using Notebooks

If you prefer using the notebooks as originally created and used (trained), you should run them in the
following environment:
- **Python Version**: 3.10.14
- **TensorFlow Version**: 2.10.0
- **TensorFlow Keras Version**: 2.10.0

These notebooks also run in Google Colab.

To set up a Jupyter Lab or Jupyter Notebook environment in Anaconda, create a new environment using the
following commands in Anaconda Command Promt or Power Shell:
```
conda create -n py310 python=3.10
conda activate py310
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python -m pip install "tensorflow==2.10"
python -m pip install pyarrow
python -m pip install fastparquet
```
This installation ensures the possibility of using the GPU.

## Usage

The application is self-intuitive and has a familiar look to other chatbots. Enter a message in the input field and press 'Enter' or click the 'Send' button. The chatbot will respond accordingly. Additionally, there are fields on the right for pre-prepared text input samples. You can press the 'Random' button to get a random sample in the input field. To exit, type 'exit' or close the window.

### Screenshot

Below is a screenshot of the app GUI window in action:

![Screenshot](images/08_app_screenshot.png)

---

