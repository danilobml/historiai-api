# Summarizing
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
# Document QA Search
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
# Location QA Search
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage

import sys
import dotenv

sys.setrecursionlimit(1000000)

dotenv.load_dotenv()


def get_text_chunks_as_docs(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def get_text_chunks_as_docs_recursive(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=100,
                                                   separators=['\n\n', '\n',
                                                               ' '])
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def generate_summary(text_input):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
    chain = load_summarize_chain(llm, chain_type="stuff")
    docs = get_text_chunks_as_docs(text_input)
    result = chain.run(docs)
    return result


def get_text_analysis(text_input, question):
    docs = get_text_chunks_as_docs_recursive(text_input)
    embeddings = OpenAIEmbeddings()
    persist_directory = 'docs/chroma/'
    store = Chroma.from_documents(persist_directory=persist_directory,
                                  embedding=embeddings,
                                  documents=docs)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=store.as_retriever())
    result = qa_chain.run({"query": question})
    return result


def get_text_locations(text_input, pattern):
    chat_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "You are a helpful academic assistant that"
                    " locates the parts of the text ({text}) that match the"
                    " {patern} given."
                    "You return the title of the document (if available), the "
                    " locations of those parts (page, if available, and line),"
                    " and the contents of those parts found."
                )
            ),
            HumanMessagePromptTemplate.from_template("Please locate the parts "
                                                     " of the docs that match "
                                                     " {pattern} in {text}"),
        ]
    )

    # docs = get_text_chunks_as_docs_recursive(text_input)
    # embeddings = OpenAIEmbeddings()
    # persist_directory = 'docs/chroma/'
    # store = Chroma.from_documents(persist_directory=persist_directory,
    #                               embedding=embeddings,
    #                               documents=docs)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    result = llm(chat_template.format_messages(text=text_input,
                                               pattern=pattern)).content

    return result
