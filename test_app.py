import os
import dotenv
import time
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, after_log, retry_if_exception_type
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type, wait_fixed
import streamlit as st
import uuid
import logging
import csv
import pandas as pd
from io import StringIO
from backoff import expo
import json
import re
from openai import OpenAI
from google.api_core.exceptions import ResourceExhausted
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
import re
# import json
# import time
# from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain.chains import create_history_aware_retriever, create_stuff_documents_chain, create_retrieval_chain
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage
from chromadb import PersistentClient
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
dotenv.load_dotenv()

chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="my_collection")

os.environ["USER_AGENT"] = "myagent"
DB_DOCS_LIMIT = 40

XAI_API_KEY = os.getenv("XAI_API_KEY")
client = OpenAI(
    api_key=XAI_API_KEY,
    base_url="https://api.x.ai/v1",
)
google_api_key = os.environ["GOOGLE_API_KEY"]
LLM_CALL_DELAY = int(os.environ.get("LLM_CALL_DELAY", 1))
INITIAL_LLM_CALL_DELAY = int(os.environ.get("INITIAL_LLM_CALL_DELAY", 1))
MAX_LLM_CALL_DELAY = int(os.environ.get("MAX_LLM_CALL_DELAY", 60))  # Maximum delay of 60 seconds
BASE_DELAY = 2  # Initial delay in seconds
MAX_DELAY = 10  # Maximum delay in seconds
RETRY_ATTEMPTS = 3 #Number of retry attempts
RATE_LIMIT_DELAY = 1
last_call_time = 0

# def get_llm_response(prompt, retries=5):
#     """Gets a response from the LLM with retry logic and adaptive delay."""
#     delay = INITIAL_LLM_CALL_DELAY
#     for attempt in range(retries):
#         try:
#             llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7, google_api_key=os.environ["GOOGLE_API_KEY"])
#             response = llm([HumanMessage(content=prompt)]).content
#             return response
#         except (ConnectionError, TimeoutError) as e:
#             logger.error(f"Network error during LLM call: {e}. Retrying in {delay} seconds...")
#         except Exception as e:
#             if "ResourceExhausted" in str(e):
#                 logger.warning(f"Quota error (429) on attempt {attempt + 1}. Retrying in {delay} seconds...")
#             else:
#                 logger.exception(f"Other error during LLM call: {e}. Retrying in {delay} seconds...")

#         sleep(delay)
#         delay = min(delay * 2, MAX_LLM_CALL_DELAY)

#     logger.error(f"LLM call failed after {retries} retries.")
#     return None

# @retry(
#     wait=wait_exponential(multiplier=1, min=2, max=10),  # Exponential backoff
#     stop=stop_after_attempt(3),  # Stop after 3 attempts
#     after=after_log(logger, logging.INFO),  # Log retry attempts
#     retry=retry_if_exception_type((ConnectionError, TimeoutError)),  # Retry on these errors
#     reraise=True  # Re-raise the exception after max retries
# )
# def call_llm_service(prompt, llm):
#     """Calls the LLM service with retry logic and timeouts."""
#     try:
#         start_time = time()
#         response = llm([HumanMessage(content=prompt)]).content
#         end_time = time()
#         logger.info(f"LLM call took {end_time - start_time:.2f} seconds.")
#         return response
#     except (ConnectionError, TimeoutError) as e:
#         logger.error(f"Network error during LLM call: {e}")
#         raise  # Re-raise to trigger retry
#     except Exception as e:
#         logger.error(f"Other error during LLM call: {e}")
#         raise  # Re-raise to trigger retry

# def generate_artifacts(stored_data, user_responses):
#     """Generates artifacts in plain text format."""
#     all_text = " ".join([str(item) for item in stored_data + user_responses])

#     try:
#         # Improved Prompts for Plain Text Output
#         persona_prompt = f"""
#         Given the following information:
#         {all_text}

#         Generate PERSONAS. For each persona, provide a name, a description, and their needs. Format each persona like this:

#         **Name:** [Persona Name]
#         **Description:** [Persona Description]
#         **Needs:** [Persona Needs]

#         Separate each persona with a horizontal line (---).
#         """

#         user_story_prompt_template = """
#         Given the following information:
#         {all_text}

#         Persona:
#         {persona_text}

#         Generate USER STORIES for this persona. Format each user story as:

#         As a [user type], I want [goal], so that [benefit].

#         Separate each user story with a newline.
#         """

#         gherkin_scenario_prompt_template = """
#         Given the following information:
#         {all_text}

#         User Story:
#         {user_story_text}

#         Generate Gherkin SCENARIOS for this user story. Format each scenario as:

#         Given [precondition]
#         When [action]
#         Then [outcome]

#         Separate each scenario with a newline.
#         """

#         persona_response = get_llm_response(persona_prompt)
#         if persona_response is None:
#             return None
#         sleep(LLM_CALL_DELAY)

#         artifacts = []
#         personas_text = persona_response.split("---\n")  # Splitting the personas
#         for persona_text in personas_text:
#             if not persona_text.strip():  # Skip empty strings
#                 continue

#             user_story_prompt = user_story_prompt_template.format(all_text=all_text, persona_text=persona_text)
#             user_story_response = get_llm_response(user_story_prompt)
#             if user_story_response is None:
#                 continue
#             sleep(LLM_CALL_DELAY)

#             user_stories_text = user_story_response.split('\n')
#             for user_story_text in user_stories_text:
#                 if not user_story_text.strip():
#                     continue
#                 gherkin_scenario_prompt = gherkin_scenario_prompt_template.format(all_text=all_text, user_story_text=user_story_text)
#                 gherkin_scenario_response = get_llm_response(gherkin_scenario_prompt)
#                 if gherkin_scenario_response is None:
#                     continue
#                 sleep(LLM_CALL_DELAY)

#                 artifacts.append({
#                     "persona": persona_text.strip(),
#                     "user_story": user_story_text.strip(),
#                     "gherkin_scenarios": [s.strip() for s in gherkin_scenario_response.split('\n') if s.strip()],
#                 })
#         return artifacts

#     except Exception as e:
#         logger.error(f"An unexpected error occurred: {e}")
#         st.error("An unexpected error occurred. Check logs.")
#         return None


def get_stored_data():
    try:
        if "vector_db" in st.session_state and st.session_state.vector_db:
            results = st.session_state.vector_db.get()

            if results:
                stored_data = results.get("documents", [])
                return stored_data  # Return only document text
            else:
                logger.info("No data found in the vector_db.")
                return []
        else:
            logger.error("vector_db is not initialized in the session state.")
            return []
    except Exception as e:
        logger.error(f"Error retrieving data from vector_db: {e}")
        return []



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

@retry(
    wait=wait_exponential(multiplier=BASE_DELAY, min=BASE_DELAY, max=MAX_DELAY) + wait_fixed(BASE_DELAY),
    stop=stop_after_attempt(RETRY_ATTEMPTS),
    retry=retry_if_exception_type(ResourceExhausted),
    reraise=False,  
)
def call_llm(llm, prompt):
    global last_call_time
    elapsed_time = time.time() - last_call_time

    if elapsed_time < RATE_LIMIT_DELAY:
        sleep_time = RATE_LIMIT_DELAY - elapsed_time
        logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)  

    try:
        response = llm.invoke(prompt)
        last_call_time = time.time() 
        return response
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        raise  

def generate_personas(stored_data, user_responses):
    try:
        if not stored_data and not user_responses:
            return ["No data available to generate personas."]

        if "persona_retry_in_progress" in st.session_state and st.session_state.persona_retry_in_progress:
            return ["Persona generation is retrying, please wait..."]

        user_response_texts = [response["text"] for response in user_responses]
        all_text = " ".join(stored_data + user_response_texts)

        prompt = f"""
        Given the following information:
        {all_text}

        Generate PERSONAS. For each persona, provide a name, a description, and their needs. Format each persona like this:

        ---PERSONA_SEPARATOR---
        **Name:** [Persona Name]
        **Description:** [Persona Description]
        **Needs:** [Persona Needs]
        ---PERSONA_SEPARATOR---

        """ # Example Prompt

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=os.environ["GOOGLE_API_KEY_NEW"],
            max_retries=1
        )

        persona_response = llm([HumanMessage(content=prompt)]).content

        if not isinstance(persona_response, str):
            if isinstance(persona_response, AIMessage):
                persona_response = persona_response.content
                logger.warning(f"LLM returned AIMessage. Extracting content: {persona_response}")
            else:
                try:
                    persona_response = json.dumps(persona_response)
                    logger.warning(f"LLM returned non-string response. Attempting JSON Conversion: {persona_response}")
                except TypeError as e:
                    logger.error(f"LLM returned unexpected non-string, non-JSON and non AIMessage response: {type(persona_response)}, {persona_response}, {e}")
                    persona_response = str(persona_response) # Force to string as a last resort
                except Exception as e:
                    logger.error(f"Error converting the response to string: {e}")
                    return ["LLM returned an unexpected response format. Check Logs."]

        personas = [p.strip() for p in re.split(r"---PERSONA_SEPARATOR---", persona_response) if p.strip()]
        print(f"Generated Personas: {personas}")
        return personas

    except ResourceExhausted as e:
        logger.error(f"Rate limit hit during persona generation: {e}")
        return generate_personas(stored_data, user_responses)  # Retry
    except Exception as e:
        logger.error(f"Error generating personas: {e}")
        return ["Error generating personas. Check Logs."]


def generate_user_stories(persona, stored_data, user_responses):
  """Generates user stories based on a persona, stored data, and user responses."""
  try:
    if stored_data and user_responses:
      return ["No data available to generate personas."]

    user_response_texts = [response["text"] for response in user_responses]
    all_text = " ".join(stored_data + user_response_texts)

    prompt = f"""Given the persona:
      {persona}

      And the following information:
      {all_text}

      Generate 2 user stories in the format: "As a [user type], I want [goal] so that [benefit]".
      """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=os.environ["GOOGLE_API_KEY"],max_retries=1)
    user_story_response = llm([HumanMessage(content=prompt)]).content

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

    user_response_texts = [response["text"] for response in user_responses]
    all_text = " ".join(stored_data + user_response_texts)

    prompt = f"""Given the user story:
      {user_story}

      And the following information:
      {all_text}

      Generate 2 Gherkin scenarios (Given/When/Then format) that test this user story.
      """
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7, google_api_key=os.environ["GOOGLE_API_KEY"],max_retries=1)
    gherkin_response = llm([HumanMessage(content=prompt)]).content

    # only the text content, removing potential LLM debugging info
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
        timestamp = time.time()  # Capture current time as metadata
        
        st.session_state.vector_db.add_texts(
            texts=[user_input],  # The user's input
            metadatas=[{"session_id": session_id, "timestamp": timestamp}],  # Metadata
            ids=[doc_id]  # Unique ID for the document
        )
        logger.info(f"Response saved to Chroma DB with ID: {doc_id}")
    except Exception as e:
        logger.error(f"Failed to store response: {e}")
        st.error(f"Error saving response: {str(e)}")

def get_user_responses_from_db(session_id):
    try:
        results = st.session_state.vector_db.get(where={"session_id": session_id})
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results.get("documents", []), results.get("metadatas", []))
        ]
    except Exception as e:
        logger.error(f"Error retrieving responses from Chroma DB: {e}")
        return []


def load_doc_to_db():
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    try:  # Try to open and write the file
                        with open(file_path, "wb") as file:
                            file.write(doc_file.read())

                        try: # Try to load the document
                            if doc_file.type == "application/pdf":
                                loader = PyPDFLoader(file_path)
                            elif doc_file.name.endswith(".docx"):
                                loader = Docx2txtLoader(file_path)
                            elif doc_file.type in ["text/plain", "text/markdown"]:
                                loader = TextLoader(file_path)
                            else:
                                st.warning(f"Document type {doc_file.type} not supported: {doc_file.name}")
                                continue # Skip to the next file

                            docs.extend(loader.load())
                            st.session_state.rag_sources.append(doc_file.name)
                            logger.info(f"Document {doc_file.name} loaded successfully.")  # Log success
                        except Exception as load_error:
                            st.toast(f"Error loading document {doc_file.name}: {load_error}", icon="⚠️")
                            logger.error(f"Error loading document {doc_file.name}: {load_error}")

                    except Exception as write_error:
                        st.toast(f"Error writing document {doc_file.name}: {write_error}", icon="⚠️")
                        logger.error(f"Error writing document {doc_file.name}: {write_error}")

                    finally: # Ensure file is deleted even if loading fails
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                logger.info(f"Temporary file {file_path} deleted.")
                            except Exception as delete_error:
                                logger.error(f"Error deleting temporary file {file_path}: {delete_error}")
                else:
                    st.error(f"Maximum number of documents reached ({DB_DOCS_LIMIT}).")
                    break # Exit the loop if limit reached
        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Document(s) {', '.join([doc_file for doc_file in st.session_state.rag_sources])} loaded successfully.", icon="✅")
            handle_business_analysis_conversation()
            st.session_state["documents_uploaded"] = True # set to true if document is uploaded
    else:
        st.session_state["documents_uploaded"] = False # set to false if documents are not uploaded
        
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
def initialize_vector_db(docs):
    """
    Initializes the vector database with the provided documents.

    Args:
        docs: List of documents to be added to the vector database.

    Returns:
        Vector database instance.
    """
    # Ensure session_id is initialized
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Create a unique collection name using the current time and session ID
    collection_name = f"{str(time.time()).replace('.', '')[:14]}_{st.session_state.session_id}"

    # Initialize the vector database
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name=collection_name,
        client=chroma_client,
    )

    # Manage the number of collections in memory (keep the last 20)
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
