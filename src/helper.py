from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from typing import List

load_dotenv()

def docs_loader(data): 

 loader  = PyMuPDFLoader(data)

 document = loader.load()

 return document



def minimal_docs_filter(docs: List[Document]) -> List[Document]:
   
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs
 


def text_splitter(extracted_data):
   splitter = RecursiveCharacterTextSplitter(
      chunk_size = 500,
      chunk_overlap = 20
   )

   chunks  = splitter.split_documents(extracted_data)
   return chunks



def text_embedding():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )   
    return embedding



extracted_data = docs_loader(data="data/Medical_book.pdf")
minimal_docs = minimal_docs_filter(docs=extracted_data)
chunk_data = text_splitter(extracted_data=minimal_docs)

embedding = text_embedding()

search = PineconeVectorStore.from_documents(
   embedding= embedding,
   index_name= "medical-chatbot",
   documents= chunk_data
)



# 768229077902.dkr.ecr.us-east-1.amazonaws.com/medicalbot

