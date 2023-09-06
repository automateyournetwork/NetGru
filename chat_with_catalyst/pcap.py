import os
import time
import json
import openai
import logging
from rich import print
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import JSONLoader
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
## Get Logger 
log = logging.getLogger(__name__)

print("[bright_yellow]Setup LLM[/bright_yellow]")
llm = ChatOpenAI(temperature=0.5, model="gpt-4")

print("[bright_yellow]Transform PCAP to JSON[/bright_yellow]")
os.system('tshark -r capture.pcap -T json > capture.json')

print("[bright_yellow]JSONLoader load the output[/bright_yellow]")
loader = JSONLoader(
    file_path="capture.json",
    jq_schema=".[] | ._source",
    text_content=False
    )

print("[bright_yellow]Load and split into pages[/bright_yellow]")
pages = loader.load_and_split()

print("[bright_yellow]Setup Recursive Character Text Splitter[/bright_yellow]")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len
    )

print("[bright_yellow]Split pages into documents with text splitter[/bright_yellow]")
docs = text_splitter.split_documents(pages)

print("[bright_yellow]Do Embeddings[/bright_yellow]")
embeddings = OpenAIEmbeddings()
#embeddings = HuggingFaceInstructEmbeddings(model_name="/app/instructor-xl",
#                                          model_kwargs={"device": "cuda"})

print("[bright_yellow]Store both the embeddings and the docs in ChromaDB vector store[/bright_yellow]")
vectordb = Chroma.from_documents(docs, embedding=embeddings)
vectordb.persist()

print("[bright_yellow]Setup memory, Conversational Buffer Memory, so our chat history is available to future prompts[/bright_yellow]")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

print("[bright_yellow]Setup a Conversational Retreival Chain from our private LLM, using the vector store, with K Values of 25, and the memory[/bright_yellow]")
qa = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(search_kwargs={"k": 3}), memory=memory)

print("[bright_yellow]Ask our question[/bright_yellow]")
question = "Please analyze the Packet Capture as JSON and summarize it and highlighting anything important.Be technical."

print(f"[bright_green]{question}[/bright_green]")

print("[bright_yellow]Run our Conversational Retreival Chain[/bright_yellow]")
result = qa.run(question)

print(f"[bright_magenta]{result}[/bright_magenta]")
