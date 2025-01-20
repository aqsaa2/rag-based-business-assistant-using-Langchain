import os
import dotenv
from time import time
import streamlit as st
import uuid
import logging
import streamlit as st
import csv
import io
import pandas as pd
from io import StringIO
import io
import csv

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
)

from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
from chromadb import PersistentClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_collection")

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 40


def get_stored_data():
    try:
        if "vector_db" in st.session_state and st.session_state.vector_db:
            # Fetch stored data from vector_db
            results = st.session_state.vector_db.get()
            
            if results:
                # Combine documents with their corresponding metadata
                stored_data = [
                    {"document": doc, "metadata": metadata}
                    for doc, metadata in zip(results.get("documents", []), results.get("metadatas", []))
                ]
                return stored_data
            else:
                logger.info("No data found in the vector_db.")
                return []
        else:
            logger.error("vector_db is not initialized in the session state.")
            return []
    except Exception as e:
        logger.error(f"Error retrieving data from vector_db: {e}")
        return []




def display_download_button(df):
    # Convert dataframe to CSV string
    csv = df.to_csv(index=False)

    # Provide the CSV file for download
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="personas_user_stories_scenarios.csv",
        mime="text/csv",
    )



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

def generate_personas(stored_data, user_responses):
  """Generates personas based on stored data and user responses."""
  try:
    if not stored_data and not user_responses:
      return ["No data available to generate personas."]

    # Combine all text data for persona generation
    all_text = " ".join(stored_data + user_responses)

    prompt = f"""Create distinct user personas based on the following information:
      {all_text}

      Each persona should include:
      - A name
      - A brief description of their background and goals
      - Their needs and pain points related to the information provided.
      """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=os.environ["GOOGLE_API_KEY"])
    persona_response = llm([HumanMessage(content=prompt)]).content

    # Get only the text content, removing potential LLM debugging info
    personas = persona_response.strip().split("\n\n")
    print(f"Generated Personas: {personas}")

    return personas

  except Exception as e:
    logger.error(f"Error generating personas: {e}")
    return ["Error generating personas."]

def generate_user_stories(persona, stored_data, user_responses):
  """Generates user stories based on a persona, stored data, and user responses."""
  try:
    if not stored_data and not user_responses:
      return ["No data available to generate personas."]

    all_text = " ".join(stored_data + user_responses)

    prompt = f"""Given the persona:
      {persona}

      And the following information:
      {all_text}

      Generate 2 user stories in the format: "As a [user type], I want [goal] so that [benefit]".
      """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=os.environ["GOOGLE_API_KEY"])
    user_story_response = llm([HumanMessage(content=prompt)]).content

    # Get only the text content, removing potential LLM debugging info
    user_stories = user_story_response.strip().split("\n")
    return user_stories
  except Exception as e:
    logger.error(f"Error generating user stories: {e}")
    return ["Error generating user stories"]

def generate_gherkin_scenarios(user_story, stored_data, user_responses):
  """Generates Gherkin scenarios based on a user story, stored data, and user responses."""
  try:
    if not stored_data and not user_responses:
      return ["No data available to generate Gherkin scenarios."]

    all_text = " ".join(stored_data + user_responses)

    prompt = f"""Given the user story:
      {user_story}

      And the following information:
      {all_text}

      Generate 2 Gherkin scenarios (Given/When/Then format) that test this user story.
      """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=os.environ["GOOGLE_API_KEY"])
    gherkin_response = llm([HumanMessage(content=prompt)]).content

    # Get only the text content, removing potential LLM debugging info
    gherkin_scenarios = gherkin_response.strip().split("\n")
    return gherkin_scenarios
  except Exception as e:
    logger.error(f"Error generating Gherkin scenarios: {e}")
    return ["Error generating Gherkin scenarios"]


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
    """
    Stores user input into the Chroma DB with metadata including session ID and timestamp.
    """
    try:
        doc_id = str(uuid.uuid4())  # Unique identifier for the response
        timestamp = time()  # Capture current time as metadata
        
        st.session_state.vector_db.add_texts(
            texts=[user_input],  # The user's input
            metadatas=[{"session_id": session_id, "timestamp": timestamp}],  # Metadata
            ids=[doc_id]  # Unique ID for the document
        )
        logger.info(f"Response saved to Chroma DB with ID: {doc_id}")
    except Exception as e:
        logger.error(f"Failed to store response: {e}")
        st.error(f"Error saving response: {str(e)}")

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
        
        # Check if URL is already processed
        if url not in st.session_state.get("rag_sources", []):
            if len(st.session_state.get("rag_sources", [])) < DB_DOCS_LIMIT:
                try:
                    # Load content from the URL
                    loader = WebBaseLoader(url)
                    docs.extend(loader.load())

                    # Check if the loader returned content
                    if not docs:
                        raise ValueError("No content was retrieved from the URL. Please provide a different URL.")

                    # Add URL to sources if successful
                    st.session_state.setdefault("rag_sources", []).append(url)

                    # Process and store documents in Chroma DB
                    _split_and_load_docs(docs)
                    st.toast(f"Document from URL {url} loaded successfully.", icon="✅")

                except ValueError as ve:
                    # Handle empty content from URL
                    st.warning(f"URL Error: {ve}")
                except Exception as e:
                    # General exception handling
                    st.error(f"Error loading document from {url}: {str(e)}. Please try a different URL.")
            else:
                st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")
        else:
            st.warning("This URL has already been processed.")




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
        You will have some context to help with your questions. Ask only relevant questions about business, business beneficiaries etc based on the context and keep them minimum (2).
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
