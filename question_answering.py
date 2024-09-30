import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch


@st.cache_resource
def load_model():
    """
    this function for loading the tokenizer and the large langauge model form Transformers
    """
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased-distilled-squad"
    )
    model = DistilBertForQuestionAnswering.from_pretrained(
        "distilbert-base-uncased-distilled-squad"
    )
    return model, tokenizer


if __name__ == "__main__":
    # Load the question-answering model and Tokenizer
    llm, tokenizer = load_model()

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
 