import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import getpass
import os
from langchain_experimental.text_splitter import SemanticChunker

# os.environ["OPENAI_API_KEY"] = getpass.getpass()

# from langchain_openai import ChatOpenAI

# model = ChatOpenAI(model="gpt-4")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/", 
               "https://lilianweng.github.io/posts/2021-09-25-train-large/", ),
            #    "https://paulgraham.com/mit.html"),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header", "post-p")
        )
    ),
)
docs = loader.load()
print(docs[1].page_content)


from langchain_core.prompts import ChatPromptTemplate

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI bot. Your name is {name}."),
    ("human", "Hello, how are you doing?"),
    ("ai", "I'm doing well, thanks!"),
    ("human", "{user_input}"),
])

prompt_value = template.invoke(
    input_variables={
        "name": "Bob",
        "user_input": "What is your name?"
    }
)

print(prompt_value)

# from langchain.embeddings import OpenAIEmbeddings, OllamaEmbeddings
# # ollembeds = OllamaEmbeddings()
# # embeddings = oaiembeds.embed_documents([x['combined_sentence'] for x in sentences])

# text_splitter = SemanticChunker(
#         buffer_size=200, 
#         add_start_index=True, 
#         breakpoint_threshold_type='standard_deviation', 
#         embedding=OpenAIEmbeddings()
#     )

# splits = []

# for i in range(len(docs)):
#     splits.extend(text_splitter.split_documents(docs[i].page_content))
#     vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# # Retrieve and generate using the relevant snippets of the blog.
# retriever = vectorstore.as_retriever()
# prompt = hub.pull("rlm/rag-prompt")


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# rag_chain.invoke("What is Task Decomposition?")