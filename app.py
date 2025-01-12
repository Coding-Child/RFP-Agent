import os
import pickle
from tqdm import tqdm
import torch

from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.document_transformers import LongContextReorder
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
os.environ["OPENAI_API_KEY"] = "your_api_key"

reordering = LongContextReorder()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    max_tokens=512,
    temperature= 0.8,
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=256,
    memory_key="history",
    return_messages=True
)


def format_docs(docs):
    docs = reordering.transform_documents(docs)

    return "\n\n".join(doc.page_content for doc in docs)


def load_memory(text):
    return memory.load_memory_variables({})['history']


class PDFIngestor:
    def __init__(
            self,
            model_name: str = 'nlpai-lab/KoE5',
            pdf_path: str = 'database/RFP',
            text_save_path: str = 'database',
            vector_store_path: str = 'database/vector_store',
        ):

        self.vector_store_path = vector_store_path
        self.pdf_path = pdf_path
        self.text_save_path = text_save_path

        if not os.path.isfile(self.text_save_path + '/rfp_data.pkl'):
            self.docs_list = self.get_docs()

            self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                model_name="gpt-4",
                chunk_size=1024,
                chunk_overlap=100
            )
            doc_splits = self.text_splitter.split_documents(self.docs_list)

            with open(f'{self.text_save_path}/rfp_data.pkl', 'wb') as f:
                pickle.dump(doc_splits, f)
        else:
            with open(f'{self.text_save_path}/rfp_data.pkl', 'rb') as f:
                doc_splits = pickle.load(f)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda:0"} if torch.cuda.is_available() else {"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        if os.path.exists(self.vector_store_path) and self.vector_store_path is not None:
            self.vector_store = FAISS.load_local(
                self.vector_store_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            self.vector_store = FAISS.from_documents(
                documents=doc_splits,
                embedding=self.embeddings,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
            )

            self.vector_store.save_local(self.vector_store_path)

    def get_docs(self):
        docs_list = list()

        if os.path.isdir(self.pdf_path):
            pdf_files = [file_name for file_name in os.listdir(self.pdf_path) if file_name.endswith(".pdf")]
            for file_name in tqdm(pdf_files, desc="Loading PDF files", unit="file", ncols=150):
                pdf_file_path = os.path.join(self.pdf_path, file_name)
                docs_list.append(PDFPlumberLoader(pdf_file_path).load())
                
        documents_list = [item for sublist in docs_list for item in sublist]
        
        return documents_list
    
    def get_retriever(self, top_k=10):
        return self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": top_k})


class Chain():
    def __init__(self, retriever):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "당신은 RFP작성에 도움을 주는 어시스턴트입니다. 아래의 예시 RFP를 참고하여, RFP를 작성해주세요."),
                MessagesPlaceholder(variable_name="history"),
                ("human", "[예시 RFP]\n{context}\n\n{query}")
            ]
        )

        self.chain = (
            {
                'context': retriever | format_docs,
                'query': RunnablePassthrough()
            }
            | RunnablePassthrough.assign(history=load_memory)
            | prompt
            | llm
            | StrOutputParser()
        )

    def astream(self, query):
        result = ''
        for chunk in self.chain.stream(query):
            print(chunk, end="", flush=True)
            result += chunk

        memory.save_context(
            {"input": query},
            {"output": result},
        )


def main():
    pdf_ingestor = PDFIngestor()
    retriever = pdf_ingestor.get_retriever()

    chain = Chain(retriever)

    print("RFP Assistant 시작! 'exit'을 입력하면 종료합니다.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("대화를 종료합니다.")
            break

        print("AI: ", end="")
        chain.astream(user_input)
        print()


if __name__ == "__main__":
    main()
