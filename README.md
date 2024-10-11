# 📚 **QA Streamlit App with Hugging Face Models**

This project is a **Question Answering (QA) application** that leverages Hugging Face's pre-trained models to extract answers from a given text. The app provides an interactive web interface built using **Streamlit**, allowing users to input paragraphs or documents and ask specific questions based on that text. The app is deployed using **Streamlit Cloud** and is continuously integrated with **GitHub Actions** for seamless deployment upon code changes.

---

## 🚀 **Features**

- **Question Answering**: Users can input a paragraph or article, ask a specific question, and get the most relevant answer based on the input text using a pre-trained QA model.
- **User-Friendly Interface**: The app provides a simple, intuitive web interface built with Streamlit. Users can input text, ask questions, and view answers easily.
- **Automated CI/CD**: GitHub Actions is used for continuous integration and deployment. Every code change pushes updates to the live app automatically.
- **Streamlit Cloud Deployment**: The app is hosted on Streamlit Cloud, ensuring that it's accessible anywhere online.

---

## 🛠 **Technology Stack**

- **Python**: Main programming language for the application.
- **Hugging Face Transformers**: Pre-trained models for question-answering tasks.
- **Streamlit**: Framework for building and deploying the interactive web app.
- **GitHub Actions**: Automating continuous integration and deployment pipelines.
- **Streamlit Cloud**: Hosting platform for deployment.

---

## 📑 **Table of Contents**
1. [Demo](#-demo)
2. [Installation](#️-installation)
3. [Usage](#️-usage)
4. [How it Works](#️-how-it-works)
5. [Deployment](#️-deployment)
---

## 🎥 **Demo**

You can try the live app here: [Streamlit QA App Demo](https://ahmedsaleh17-huggingface-question-ans-question-answering-y32ihl.streamlit.app/)

---

## 🛠️ **Installation**

### Prerequisites

Ensure you have the following installed:

- **Python 3.10**
- **Pip** (Python package installer)

### Step-by-Step Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ahmedsaleh17/huggingface-question-answering-app
   code huggingface-question-answering-app
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   conda create -p qa-env python==3.10 -y
   conda activate qa-env/
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app locally**:
   ```bash
   streamlit run question_answering.py
   ```

   The app will be running at `http://localhost:8501/`.

---

## 🖥️ **Usage**

1. **Input a Paragraph**: Paste or type a paragraph or document into the text input area.
2. **Ask a Question**: In the question input field, type a question related to the input text.
3. **Get an Answer**: Click on the "Get Answer" button, and the app will display the most relevant answer extracted from the provided text.

---

## ⚙️ **How It Works**

1. **Question Answering Model**:
   - The app utilizes Hugging Face’s `transformers` library with a pre-trained PyTorch model, specifically `distilbert-base-uncased-distilled-squad`, for question answering.
   - The model is loaded using `DistilBertTokenizer` to tokenize the input text and `DistilBertForQuestionAnswering` to predict the most relevant answer span.
   - **PyTorch** is used as the backend for model inference.

   Here’s how the model is loaded and used:

   ```python
   from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
   import torch

   # Load the pre-trained tokenizer and model for QA
   tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
   model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')



   question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = torch.argmax(outputs.start_logits)
    answer_end_index = torch.argmax(outputs.end_logits)

    predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    tokenizer.decode(predict_answer_tokens)
    ```

2. **Streamlit Web App**:
   - **Streamlit** provides a simple way to create an interactive web interface.
   - Users input the text and question, and the app fetches the answer by passing the input to the Hugging Face model.
   
   Example Streamlit structure:
   ```python
   import streamlit as st
   from transformers import pipeline

    # Initialize our streamlit app

    # set page title
    st.header("HuggingFace Application")
    # streamlit UI
    st.title("Question Answering using distilbert LLM 🤖")

    # user input for question and context

    context = st.text_area("Enter the Context:", height=200)
    question = st.text_area("Enter your question:")

    # Get answer buttom
    if st.button("Get Answer"):
        if context and question:

            # pass inputs to tokenizer
            inputs = tokenizer(question, context, return_tensors="pt")
            with torch.no_grad():
                # get the output
                output = llm(**inputs)

            answer_start_index = torch.argmax(output.start_logits)
            answer_end_index = torch.argmax(output.end_logits)

            # Get final response 
            predict_answer_tokens = inputs.input_ids[
                0, answer_start_index : answer_end_index + 1
            ]
            result = tokenizer.decode(predict_answer_tokens)

            st.write("Answer:", result)

        else:
            st.write("Please enter both context and question.")
 
   ```

3. **CI/CD Pipeline**:
   - **GitHub Actions** is used to automatically build and deploy the app whenever changes are pushed to the main branch.
   - The `.github/workflows/deploy.yml` file handles the CI/CD pipeline by installing dependencies, running tests, and deploying the app on Streamlit Cloud.

   Example GitHub Actions Workflow:
   ```yaml
    name: Deploy Streamlit App
    on:
      push:
        branches:
          - main

    jobs:
      build:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v3
          - name: Set up Python environment
            uses: actions/setup-python@v4
            with:
              python-version: '3.10'

          - name: Install dependencies
            run: |
              pip install -r requirements.txt

          - name: Build Streamlit app
            run: |
              nohup streamlit run question_answering.py --server.port 8501 --server.headless True &
              sleep 10  # Wait for a few seconds to allow the app to start

          - name: Check if Streamlit app is running
            run: |
              if curl -s --head  --request GET http://localhost:8501 | grep "200 OK" > /dev/null; then 
                echo "Streamlit app is running."
              else
                echo "Streamlit app is not running."
                exit 1
              fi
   ```

---

## ☁️ **Deployment**

This app is deployed on **Streamlit Cloud**. Follow these steps to deploy it yourself:

1. **Sign up for Streamlit Cloud**: [Streamlit Cloud](https://streamlit.io/cloud)
2. **Connect your GitHub Repository**: Link your repository to Streamlit Cloud.
3. **Deploy**: Once linked, your app will automatically deploy, and any future updates to your repository will trigger a new deployment.

Ensure you have a valid `requirements.txt` file with the following dependencies:
```plaintext
streamlit
transformers
torch  # Ensure PyTorch is included as a dependency for model inference
```

---


## 🛡️ **Acknowledgements**

- **Hugging Face** for providing the amazing NLP models.
- **Streamlit** for offering a simple and powerful way to build and deploy web apps.
- **GitHub Actions** for automating CI/CD pipelines.

---

Feel free to reach out if you have any questions or suggestions!