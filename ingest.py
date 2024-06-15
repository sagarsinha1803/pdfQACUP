from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os
from constants import CHROMA_DB_SETTINGS

persist_directory = 'db'

def main():
    for root, dir, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))

    douments = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(douments)

    #creating embedding
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    #create vector store
    db = Chroma.from_documents(texts, embedding=embedding, persist_directory=persist_directory, client_settings=CHROMA_DB_SETTINGS)
    db.persist()
    db=None


if __name__ == '__main__':
    main()