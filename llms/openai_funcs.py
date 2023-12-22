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
    store = Chroma(persist_directory=persist_directory,
                   embedding_function=embeddings)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm,
                                           retriever=store.as_retriever())
    result = qa_chain({"inputs": docs, "query": question})
    print(result)
