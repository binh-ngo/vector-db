import os
import pinecone
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from dotenv import load_dotenv

from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
load_dotenv()

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment="gcp-starter")

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader("/Users/binhngo/code/vector-db/mediumblogs/mediumblog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    qa = RetrievalQA.from_chain_type(
      llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
    )

    query = "What is Bedrock? Give me a 15 word answer for a beginner"
    result = qa({"query": query})
    print(result)
