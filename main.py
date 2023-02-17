import os

import gspread
import numpy as np
import openai
import pandas as pd
import streamlit as st
import tiktoken
from PIL import Image
from openai.embeddings_utils import distances_from_embeddings

SCOPES = ['https://www.googleapis.com/auth/spreadsheets',
          'https://www.googleapis.com/auth/drive',
          'https://www.googleapis.com/auth/documents.readonly']

# Define root domain to crawl
HTTP_URL_PATTERN = r'^http[s]*://.+'
max_tokens = 500
trueloaderimage = Image.open('DOM.png')
donracksimage = Image.open('DOM.png')
api_key = os.environ.get("DIGITAL_SEO_KEY")
openai.api_key=api_key

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens=max_tokens):
    # Split the text into sentences
    sentences = text.split('. ')
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined tpip install ogether in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks

def getopenairesponse(domain, userquestion):
    # Create a list to store the text files
    texts = []

    # Get all the text files in the text directory
    for file in os.listdir("text/" + domain + "/"):
        # Open the file and read the text
        with open("text/" + domain + "/" + file, "r") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
            texts.append((file[11:-4].replace('-', ' ').replace('_', ' ').replace('#update', ''), text))

    ######################################

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns=['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv('processed/scraped.csv')
    df.head()

    ################################
    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv('processed/scraped.csv', index_col=0)
    df.columns = ['title', 'text']

    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    # Visualize the distribution of the number of tokens per row using a histogram
    df.n_tokens.hist()

    ################################
    shortened = []
    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append(row[1]['text'])

    df = pd.DataFrame(shortened, columns=['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()

    ################################
    df['embeddings'] = df.text.apply(
        lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    df.to_csv('processed/embeddings.csv')
    df.head()

    df = pd.read_csv('processed/embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    df.head()
    #return("Your Answer\n" + str(answer_question(df, question=userquestion, debug=True)))
    return "I Dont Know"

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

def create_context(question, df, max_len=1800, size="ada"):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')

    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
        df,
        model="text-davinci-003",
        question="What is annual cap for 01 Visa",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=150,
        stop_sequence=None):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        #print("LENGTH OF PROMPT:\n" + len(context))
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            #prompt=f"Assume you are Chat Support Executive . Answers need to be short , assertive,friendly and provoke conversation. Frame the answer for the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            prompt=f"Assume you are Chat Support Executive. Answers should be easy to understand, concise (up to 50 words) and helpful. Keep it relevant to the question. Frame the answer for the question based on the context below, and if the question can't be answered based on the context, say \"i don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        #print("Model used  :: " + str(response["model"]))
        #print("prompt_tokens :::  " + str(response["usage"][0]["prompt_tokens"].strip()))
        #print("completion_tokens :::  " + str(response["usage"][0]["completion_tokens"].strip())
        #print("total_tokens :::  " + str(response["usage"][0]["total_tokens"].strip()))
        print(response)
        return response["choices"][0]["text"].strip()

    except Exception as e:
        print(e)
        print("Errr while processing name" )
        return ""

###############################################################################

st.header("Welcome to Chat Support Help Centre")
st.info('This tool will be used by Chat Support executives when the question posted by customer requires a detailed and indepth answer. \n The tool has been pre-trained with information about the business offering ', icon="ℹ️")
st.caption("Last updated information fed to this tool : Dec 2022.")
#tabimage = st.sidebar.image("DOM.png", width=200)
st.header("How to Use")
st.write("Step1 : Select the Business name from the sidebar")
st.write("Step2: Enter your question")
st.write("Step3: Click Get Answers button")
st.write("Step4: Wait for few seconds to get your answer")

st.caption("\n\n If you face any issues, send the screenshot to deepa@digitalseo.in")


#name = st.sidebar.text_input("Enter your name", 'Deepa')
#question = st.text_input("What's Your Question", key="myquestion")

#if st.button('Get Answer'):
#    answer = getopenairesponse("www.donracks.co.in", question)
#    #answer = " The answer generated by application wll be displayed hereThe \n answer generated by application wll be displayed hereThe answer generated by application wll be displayed hereThe answer generated by application wll be displayed hereThe answer generated by application wll be displayed hereThe answer generated by application wll be displayed hereThe answer generated by application wll be displayed hereThe answer generated by application wll be displayed here"
#    timeanswered = datetime.datetime.now()
#    st.header(f"\n  Your Answer ")

    #st.write(answer)
    #st.write(timeanswered)
#    st.subheader(answer)
#    st.caption(timeanswered)

#if st.button('Clear'):
#    question = ''
#    st.write(f"\n  Cleared Results !")




###################

