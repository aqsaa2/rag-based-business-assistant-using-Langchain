import os
import dotenv
from time import time
import streamlit as st
import uuid
import logging
import streamlit as st
import csv
import io
import io
import csv
import logging
import streamlit as st
from io import StringIO

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)

# pip install docx2txt, pypdf
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chromadb import PersistentClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_collection")

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 40


# Create a collection
# collection_name = chroma_client.create_collection(
#     name="chat_pdf_agent", 
#     metadata={"hnsw:space": "cosine"}
# )

# collection = chroma_client.get_collection("chat_pdf_agent")


# def display_stored_data():
#     try:
#         # Retrieve all data from the Chroma collection
#         results = collection.get()
        
#         if results:
#             # Display documents and their metadata
#             st.subheader("Stored Data in Vector Store:")
#             for idx, doc in enumerate(results["documents"]):
#                 metadata = results["metadatas"][idx]
#                 st.write(f"**Document {idx + 1}:** {doc}")
#                 st.write(f"**Metadata:** {metadata}")
#                 st.divider()
#         else:
#             st.write("No data stored in the vector store yet.")
#     except Exception as e:
#         logger.error(f"Failed to fetch data: {e}")
#         st.error(f"Error fetching data: {str(e)}")

# def generate_csv(personas, user_stories, gherkin_scenarios):
#     # Combine data into a single dataframe
#     data = {
#         "Category": ["Persona"] * len(personas) + ["User Story"] * len(user_stories) + ["Gherkin Scenario"] * len(gherkin_scenarios),
#         "Content": personas + user_stories + gherkin_scenarios
#     }
#     df = pd.DataFrame(data)

#     # Create a CSV in memory
#     csv_buffer = StringIO()
#     df.to_csv(csv_buffer, index=False)
#     csv_buffer.seek(0)
#     return csv_buffer.getvalue()

# def download_csv_button(personas, user_stories, gherkin_scenarios):
#     # Generate the CSV file content
#     csv_content = generate_csv(personas, user_stories, gherkin_scenarios)

#     # Create a download button
#     st.download_button(
#         label="Download Outputs as CSV",
#         data=csv_content,
#         file_name="outputs.csv",
#         mime="text/csv"
#     )

# def csv_output(llm, stored_data):
#     personas = llm(generate_persona_prompt(stored_data))
#     user_stories = llm(generate_user_story_prompt(stored_data))
#     gherkin_scenarios = llm(generate_gherkin_scenario_prompt(stored_data))

#     download_csv_button(personas, user_stories, gherkin_scenarios)

def get_stored_data():
    """Fetch stored data from Chroma vector database."""
    try:
        documents = st.session_state.vector_db.get()  
        if not documents or "documents" not in documents:
            logger.warning("No documents found in the vector database.")
            return []
        return documents["documents"]
    except Exception as e:
        logger.error(f"Failed to fetch stored data: {e}")
        return []

def generate_csv_from_chroma(llm, stored_data):
    """Generate CSV content for personas, user stories, and business scenarios."""
    personas = llm(generate_persona_prompt(stored_data))
    user_stories = llm(generate_user_story_prompt(stored_data))
    gherkin_scenarios = llm(generate_gherkin_scenario_prompt(stored_data))

    # Create a CSV in memory
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["Type", "Name/Story/Scenario", "Description"])
    writer.writeheader()

    # Add personas
    for persona in personas:
        writer.writerow({
            "Type": "Persona",
            "Name/Story/Scenario": persona.get("Name", "Unknown Persona"),
            "Description": persona.get("Description", ""),
        })

    # Add user stories
    for story in user_stories:
        writer.writerow({
            "Type": "User Story",
            "Name/Story/Scenario": story.get("Story", "Unknown Story"),
            "Description": story.get("Reason", ""),
        })

    # Add Gherkin scenarios
    for scenario in gherkin_scenarios:
        writer.writerow({
            "Type": "Business Scenario",
            "Name/Story/Scenario": scenario.get("Scenario", "Unknown Scenario"),
            "Description": scenario.get("Description", ""),
        })

    # Prepare CSV data for download
    output.seek(0)
    return output.getvalue()

def display_download_button(llm, stored_data):
    """Generate CSV file and provide download button."""
    if not stored_data:
        st.warning("No data available to generate the CSV.")
        return

    csv_data = generate_csv_from_chroma(llm, stored_data)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="personas_user_stories.csv",
        mime="text/csv",
    )


def generate_persona_prompt(stored_data):
    return f"""
    Based on the following stored information:
    {stored_data}
    
    Generate a detailed persona including:
    - Name
    - Role/Job Title
    - Background
    - Goals
    - Pain Points
    """

def generate_user_story_prompt(stored_data):
    return f"""
    Based on the following stored information:
    {stored_data}
    
    Generate user stories in the format:
    "As a [user type], I want [goal] so that [reason]."
    """

def generate_gherkin_scenario_prompt(stored_data):
    return f"""
    Based on the following stored information:
    {stored_data}
    
    Generate Gherkin scenarios in the format:
    ```
    Feature: [Feature Name]
      Scenario: [Scenario Name]
        Given [initial context]
        When [event occurs]
        Then [expected outcome]
    ```
    """

# def generate_outputs(llm, stored_data):
#     personas = llm(generate_persona_prompt(stored_data))
#     user_stories = llm(generate_user_story_prompt(stored_data))
#     gherkin_scenarios = llm(generate_gherkin_scenario_prompt(stored_data))
#     return personas, user_stories, gherkin_scenarios

# //////////// JUST TO CHECK WHETHER USER RESPONSES ARE BEING STORED OR NOT ////////////
def get_documents_from_collection():
    if st.session_state.vector_db:
        documents = st.session_state.vector_db.get()  
        return documents['documents']  
    else:
        return []


# Function to retrieve user responses from vector_db collection
def get_user_responses():
    if st.session_state.vector_db:
        responses = st.session_state.vector_db.get()  
        return responses['documents']  
    else:
        return []

# Function to display all documents and user responses
def display_all_chroma_data():
    try:
        # Retrieve all data from the Chroma collection
        if st.session_state.vector_db:
            # Fetch all documents and metadata from the collection
            results = st.session_state.vector_db.get()
            
            if results:
                # Split the results into respective types
                documents = results.get("documents", [])
                metadatas = results.get("metadatas", [])

                # Display the stored data
                st.subheader("Stored Data in Chroma DB:")

                for idx, doc in enumerate(documents):
                    metadata = metadatas[idx] if idx < len(metadatas) else {}
                    
                    # Check metadata to identify the type of data (e.g., user response, scraped URL, or document content)
                    data_type = metadata.get("type", "Unknown")
                    st.write(f"**Data Type:** {data_type}")
                    st.write(f"**Content:** {doc}")
                    st.write(f"**Metadata:** {metadata}")
                    st.divider()

            else:
                st.write("No data stored in the Chroma DB yet.")
        else:
            st.error("Collection not found.")
    except Exception as e:
        logger.error(f"Failed to fetch stored data: {e}")
        st.error(f"Error fetching stored data: {str(e)}")

# /////////////////////////////////////////////////////////////////////
        

def initialize_business_questions():
    questions = [
        "What is your business's industry?",
        "What is your business website?",
        "What government entities govern and regulate your core business industry?",
        "Who are the beneficiaries of your business?",
        "What are the services or products you deliver to your beneficiaries?"
    ]
    return questions


    
def handle_business_analysis_conversation():
    if "business_questions" not in st.session_state:
        st.session_state.business_questions = initialize_business_questions()
        st.session_state.business_answers = []

    
    if st.session_state.business_questions:
        next_question = st.session_state.business_questions.pop(0)
        st.session_state.messages.append({"role": "assistant", "content": next_question})

    elif len(st.session_state.business_answers) == 5:
        st.session_state.messages.append({"role": "assistant", "content": "Thank you for providing the information!"})
        st.session_state.business_analysis_completed = True


# Function to stream the response of the LLM 
def stream_llm_response(llm_stream, messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# --- Indexing Phase ---

# Function to store responses in Chroma DB
def store_user_responses_to_db(user_input, session_id):
    try:
        doc_id = str(uuid.uuid4())
        st.session_state.vector_db.add(
            documents=[user_input],
            metadatas=[{"session_id": session_id}],
            ids=[doc_id]
        )
        logger.info(f"Response saved to Chroma DB with ID: {doc_id}")
    except Exception as e:
        logger.error(f"Failed to store response: {e}")

def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = [] 
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(file_path)
                        elif doc_file.name.endswith(".docx"):
                            loader = Docx2txtLoader(file_path)
                        elif doc_file.type in ["text/plain", "text/markdown"]:
                            loader = TextLoader(file_path)
                        else:
                            st.warning(f"Document type {doc_file.type} not supported.")
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)

                    except Exception as e:
                        st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                        print(f"Error loading document {doc_file.name}: {e}")
                    
                    finally:
                        os.remove(file_path)

                else:
                    st.error(F"Maximum number of documents reached ({DB_DOCS_LIMIT}).")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document *{str([doc_file.name for doc_file in st.session_state.rag_docs])[1:-1]}* loaded successfully.", icon="✅")
            handle_business_analysis_conversation()

def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.get("rag_sources", []):
            if len(st.session_state.get("rag_sources", [])) < 10:
                try:
                    loader = WebBaseLoader(url)  # Scraping the URL content
                    docs.extend(loader.load())  # Load the scraped content as documents.
                    st.session_state.setdefault("rag_sources", []).append(url)  # Track the scraped URL.

                except Exception as e:
                    st.error(f"Error loading document from {url}: {e}")

                if docs:
                    _split_and_load_docs(docs)  # Process and store the scraped content in Chroma DB.
                    st.toast(f"Document from URL {url} loaded successfully.", icon="✅")
            else:
                st.error("Maximum number of documents reached (10).")


# Update this part in the initialize_vector_db function:
def initialize_vector_db(docs):
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=f"{str(time()).replace('.', '')[:14]}_" + st.session_state['session_id'],
    )

    # We need to manage the number of collections that we have in memory, we will keep the last 20
    chroma_client = vector_db._client
    collection_names = sorted([collection.name for collection in chroma_client.list_collections()])
    print("Number of collections:", len(collection_names))
    while len(collection_names) > 20:
        chroma_client.delete_collection(collection_names[0])
        collection_names.pop(0)

    return vector_db



def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
    )

    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)






# --- Retrieval Augmented Generation (RAG) Phase ---

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages."),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """You are a helpful business analyst. You will have to ask questions from user.
        You will have some context to help with your questions. Ask only relevant questions about business, business beneficiaries etc and keep them minimum (2).
        You can also use your knowledge to assist answering the user's queries.\n
        {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(llm_stream, messages):
    print("This is RAG response")
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response_message = "(RAG Response)\n"
    for chunk in conversation_rag_chain.pick("answer").stream({"messages": messages[:-1], "input": messages[-1].content}):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})