# Summarizing
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document
# Document QA Search
# from langchain.llms.openai import OpenAI
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores.chroma import Chroma
# from langchain.chains import RetrievalQA

import dotenv

dotenv.load_dotenv()


def get_text_chunks_as_docs(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def generate_summary(text_input):
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
    chain = load_summarize_chain(llm, chain_type="stuff")
    docs = get_text_chunks_as_docs(text_input)
    result = chain.run(docs)
    return result


# def get_text_analysis(question):
#     loader = TextLoader("data/marx_manifesto.txt")
#     docs = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
#                                                    chunk_overlap=100,
#                                                    separators=['\n\n', '\n',
#                                                                ' ', ''])
#     texts = text_splitter.split_documents(docs)
#     embeddings = OpenAIEmbeddings()
#     store = Chroma(texts, embeddings, collection_name="marx-manifesto")
#     llm = OpenAI(temperature=0)
#     chain = RetrievalQA.from_chain_type(llm, retriever=store.as_retriever())

#     print(chain.run(question))


# get_text_analysis("What did Marx say about France?")
