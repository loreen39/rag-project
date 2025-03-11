import gradio as gr
import os
import json
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage

import json
from dotenv import load_dotenv

# load API-KEYS
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY")

# Load prompt templates from JSON file
with open("prompts.json", "r") as file:
    data = json.load(file)
#Extract the prompt
astronomy_system_prompt = data.get("astronomy_system_prompt", "")


# embeddings models
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", 
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Define paths
db_path = "VectorDB"
data_path = "Astronomy_Books"


# Check if the vector store exists
if os.path.exists(db_path) and os.listdir(db_path):
    vector_store = FAISS.load_local(db_path, huggingface_embeddings, allow_dangerous_deserialization=True)
else:
    # Load documents from PDFs
    loader = PyPDFDirectoryLoader(data_path)
    pages = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    splitted_pages = text_splitter.split_documents(pages)

    # Create and save the vector store
    vector_store = FAISS.from_documents(splitted_pages, huggingface_embeddings)
    vector_store.save_local(db_path)



# define retriever
retriever = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 3, "score_threshold": 0.5}
)

# define llm
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)


astronomy_prompt = ChatPromptTemplate.from_messages(
    [("system", astronomy_system_prompt), MessagesPlaceholder("chat_history"),  ("human", "{input}"),  ]
)

question_answer_chain = create_stuff_documents_chain(llm, astronomy_prompt)

rag_chain = create_retrieval_chain(retriever , question_answer_chain)

def chat_with_model(history, new_message, chat_history):
    # Invoke the retrieval-augmented generation (RAG) model
    response = rag_chain.invoke({"input": new_message, "chat_history": chat_history})

    # Extract the assistant's response
    assistant_message = response["answer"]

    # Update chat history
    chat_history.append(HumanMessage(content=new_message))
    chat_history.append(response["answer"])

    # Update displayed chat history (Gradio UI)
    history.append((new_message, assistant_message))

    return history, ""


def gradio_chat_app():
    with gr.Blocks(css="""
        body {
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .chatbot-container {
            padding: 20px;
            border-radius: 15px;
            margin: auto;
        }
        .title {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            color: #FFD700;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            margin-bottom: 15px;
        }
    """) as app:
        chat_history = gr.State([])
        
        gr.HTML("<h1>🚀 AstroBot: Your Astronomy Assistant 🔭</h1>")  # ✅ Replaced gr.Markdown()
        gr.Markdown("🌌 **Ask me anything about space, black holes, exoplanets, and galaxies!** 🌠")

        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label="AstroBot Chat Interface", elem_classes=["chatbot-container"])
                user_input = gr.Textbox(label="Your Message", placeholder="Type something...", lines=1)
                send_button = gr.Button("🚀 Send")
                clear_button = gr.Button("🛸 Clear Chat")

        def clear_chat():
            return [], "", []

        send_button.click(fn=chat_with_model, inputs=[chatbot, user_input, chat_history], outputs=[chatbot, user_input])
        clear_button.click(fn=clear_chat, inputs=[], outputs=[chatbot, user_input, chat_history])

    return app

if __name__ == "__main__":
    app = gradio_chat_app()
    app.launch(share=True)
