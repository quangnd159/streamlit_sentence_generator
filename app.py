
import os

import streamlit as st
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chat_models import ChatOpenAI
import azure.cognitiveservices.speech as speechsdk

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
# eleven_api_key = os.getenv("ELEVEN_API_KEY")

llm = ChatOpenAI(temperature=1.2, model="gpt-3.5-turbo-0613")


# def generate_answer(card, topic):
#     """Generate an answer using the langchain library and OpenAI's GPT-3 model."""
#     prompt = PromptTemplate(
#         input_variables=["card", "topic"],
#         template=""" 
#          You are an 18-year-old girl who is attending an English test. 
#          Answer the IELTS Speaking Part 2 task card in 200 words using the chosen topic. Use a conversational tone but not too casual. The vocabulary should be that of a high school student.
#          Avoid formality and avoid written English such as furthermore, therefore, overall, and in conclusion. 
#          Here is the task card: {card}. And here's the chosen topic: {topic}
#                  """
#     )
#     chain = LLMChain(llm=llm, prompt=prompt)
#     return chain.run({
#     'card': card,
#     'topic': topic
#     })

def text_to_speech(text):
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_SUBSCRIPTION_KEY'), region=os.environ.get('SPEECH_REGION'))
    audio_config = speechsdk.audio.AudioOutputConfig(filename="audio.wav")

    # The language of the voice that speaks.
    speech_config.speech_synthesis_voice_name='en-US-SaraNeural'

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    # Synthesize the text to the default speaker.
    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()
    
    return speech_synthesis_result

def app():
    st.title("Sentence Generator")

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
                template = """Suggest two other vocabulary items that can serve the same purpose as the provided item. An example is provided to give you the meaning of the given item. Make sure to give suggestions appropriate for academic essays. The examples given for your suggestions must use different topics other than the one in the given example. Use nice markdown format with the output.
                Here is the vocabulary item: {item}. And here's the example using the item: {example}"""
                prompt_template = PromptTemplate(input_variables=["item", "example"], template=template)
                more_items_chain = LLMChain(llm=llm, prompt=prompt_template)
                more_items = more_items_chain.run({
                    "item": item,
                    "example": example
                })
                # audio = text_to_speech(answer_text)
                # audio_data = audio.audio_data
            st.markdown(sentence_text)
            st.header('Similar vocabulary items')
            st.markdown(more_items)
            # st.audio(audio_data, format='audio/wav')
            # st.markdown(collocations)


if __name__ == '__main__':
    app()
