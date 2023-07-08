import os

import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=1, model="gpt-3.5-turbo")

def app():
    st.set_page_config(page_title="SenGenAI", page_icon="⚡")
    st.title("SenGenAI ⚡")
    st.markdown("## A sentence generator powered by AI")
    st.sidebar.markdown(
        """
        ### 💡 Instructions 
        1. Enter a vocabulary item
        2. Enter an example that uses the item, with a clear context
        3. Hit **Submit**
        """
    )
    st.sidebar.info("You will get more examples using the given vocabulary item in different contexts, along with two more words or phrases that help you achieve the same (or similar) purpose.")
    st.sidebar.success("🌱 This app aims to help you expand writing vocabulary by providing you with good examples and similar expressions, which you can put in an intelligent flashcard system like Anki.")

    with st.form(key='my_form'):
        item = st.text_input("Enter a vocabulary item")
        example = st.text_area(
            "Enter an example",
            max_chars=None,
            placeholder="Enter an example of the vocabulary item above",
            height=100,
        )

        if st.form_submit_button("Submit"):
            with st.spinner('Generating sentences...'):
                # Chain 1: Generating an answer
                template = """Use the provided vocabulary item and write three examples using various topics typically found in an IELTS writing task 2. An example is provided to give you the context of how the item should be used.
                Here is the vocabulary item: {item}. And here's the example using the item: {example}"""
                prompt_template = PromptTemplate(input_variables=["item", "example"], template=template)
                sentence_chain = LLMChain(llm=llm, prompt=prompt_template)
                sentence_text = sentence_chain.run({
                    "item": item,
                    "example": example
                })
                # Chain 2: Extract collocations from answer
                template = """Suggest two other common vocabulary items that can serve the same purpose as the provided item. An example is provided to give you the meaning of the given item. Make sure to give suggestions appropriate for academic essays. The examples given for your suggestions must use different topics other than the one in the given example. Use nice markdown format with the output.
                Here is the vocabulary item: {item}. And here's the example using the item: {example}"""
                prompt_template = PromptTemplate(input_variables=["item", "example"], template=template)
                more_items_chain = LLMChain(llm=llm, prompt=prompt_template)
                more_items = more_items_chain.run({
                    "item": item,
                    "example": example
                })
            st.markdown(sentence_text)
            st.header('Similar vocabulary items')
            st.markdown(more_items)
    st.write("By [Quang](https://dqnotes.com)")

if __name__ == '__main__':
    app()
