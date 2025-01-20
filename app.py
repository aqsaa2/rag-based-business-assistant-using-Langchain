import streamlit as st
import os
import dotenv
import pandas as pd
import uuid
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores.chroma import Chroma
from test_app import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
    store_user_responses_to_db,
    display_download_button,
    generate_personas,
    generate_user_stories,
    generate_gherkin_scenarios,
    display_all_chroma_data,
)
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
import chromadb
from chromadb import PersistentClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Persistent Client with path
chroma_db_path = "chroma_db"
chroma_client = PersistentClient(path=chroma_db_path)

dotenv.load_dotenv()

st.set_page_config(
    page_title="RAG-Enhanced Chat",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Header
st.header("RAG-Enhanced Chat Application")
st.subheader("Upload documents or URLs to enhance the conversation with relevant context")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    logger.info(f"New session started: {st.session_state.session_id}")

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm ready to help. You can upload documents or provide URLs to enhance our conversation with relevant context."}
    ]

# Sidebar configuration
with st.sidebar:
    # API Key Management
    if "GOOGLE_API_KEY" not in os.environ:
        with st.popover("🔐 API Key Configuration"):
            google_api_key = st.text_input(
                "Enter your Google API Key",
                value=os.getenv("GOOGLE_API_KEY", ""),
                type="password",
                key="google_api_key",
            )
            if not google_api_key:
                st.warning("Please enter your Google API Key to continue")
    else:
        google_api_key = os.environ["GOOGLE_API_KEY"]

    # Add this to your Streamlit interface, e.g., after a button click or on app load
    #button to show stored data
    # if st.button("Show Stored Data"):
    #     display_all_chroma_data()

    st.divider()
    chroma_db_path = os.path.join(os.getcwd(), "chroma_db")
    if os.path.exists(chroma_db_path) and os.listdir(chroma_db_path):
        st.session_state["chroma_db"] = "Path to chroma_db loaded"
    else:
        st.session_state["chroma_db"] = None

    # RAG Controls
    col1, col2 = st.columns(2)
    with col1:
        is_vector_db_loaded = (
            ("vector_db" in st.session_state and st.session_state.vector_db is not None) or
            ("chroma_db" in st.session_state and st.session_state.chroma_db is not None)
        )

        use_rag = st.toggle(
            "Enable RAG",
            value=is_vector_db_loaded,
            key="use_rag",
            disabled=not is_vector_db_loaded,
        )

    with col2:
        if st.button("Clear Chat", type="primary"):
            st.session_state.messages = [st.session_state.messages[0]]
            st.rerun()

    # Document Upload Section
    st.header("Knowledge Base")

    # File upload
    uploaded_files = st.file_uploader(
        "🗄 Upload Documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, DOCX, MD",
        key="rag_docs",
        on_change=load_doc_to_db
    )

    # URL input
    st.text_input(
        "🌐 Add Web Content",
        placeholder="https://example.com",
        help="Enter a URL to include web content",
        key="rag_url",
        on_change=load_url_to_db
    )

    # Show loaded sources
    with st.expander(f"Knowledge Base Sources ({len(st.session_state.rag_sources)})"):
        if st.session_state.rag_sources:
            for source in st.session_state.rag_sources:
                st.write(f"- {source}")
        else:
            st.write("No sources loaded yet")
            
    # To display the stored data in vector db
    if uploaded_files:
        if st.button("Show Stored Data"):
            display_all_chroma_data()

# for tracking context-building
if "user_input_count" not in st.session_state:
    st.session_state.user_input_count = 0

if "ask_for_csv" not in st.session_state:
    st.session_state.ask_for_csv = False

if not google_api_key and "GOOGLE_API_KEY" not in os.environ:
    st.warning("⚠️ Please configure your API key in the sidebar to continue")
else:
    try:
        llm_stream = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0.4,
            streaming=True,
            google_api_key=google_api_key,
        )

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input for answering business questions
        if prompt := st.chat_input("Type your response here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.user_input_count += 1

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                messages = [
                    HumanMessage(content=m["content"]) if m["role"] == "user"
                    else AIMessage(content=m["content"])
                    for m in st.session_state.messages
                ]

                # If RAG is disabled or vector DB is not initialized, use simple LLM response
                if not st.session_state.use_rag or "vector_db" not in st.session_state:
                    st.write_stream(stream_llm_response(llm_stream, messages))
                else:
                    # Use RAG for responses when enabled and vector DB is available
                    store_user_responses_to_db(prompt, st.session_state.session_id)  # Save user response to Chroma DB
                    st.write_stream(stream_llm_rag_response(llm_stream, messages))

            # Check if enough user input has been gathered to generate personas
            if st.session_state.user_input_count >= 5 and "asked_for_csv" not in st.session_state:
                st.session_state["asked_for_csv"] = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I have gathered enough information. Would you like me to generate a CSV file with personas, user stories, and business scenarios in Gherkin format? (Type 'yes' to confirm)"
                })
                st.rerun()

        if "asked_for_csv" in st.session_state and st.session_state["asked_for_csv"]:
            last_message = st.session_state.messages[-1]["content"]

            if last_message.strip().lower() == "yes":
                stored_data = get_stored_data()
                user_responses = get_user_responses_from_db(st.session_state.session_id)

                if stored_data:
                    # Generate Personas, User Stories, and Gherkin Scenarios
                    personas = generate_personas(stored_data, user_responses)
                    data_for_csv = []

                    for persona in personas:
                        # For each persona, generate user stories
                        user_stories = generate_user_stories(persona, stored_data, user_responses)

                        for user_story in user_stories:
                            # For each user story, generate business scenarios in Gherkin format
                            gherkin_scenarios = generate_gherkin_scenarios(user_story, stored_data, user_responses)

                            # Store the persona, user story, and gherkin scenarios in a structured format
                            for scenario in gherkin_scenarios:
                                data_for_csv.append({
                                    "Persona": persona,
                                    "User Story": user_story,
                                    "Business Scenario (Gherkin)": scenario
                                })
                    
                    
                    df = pd.DataFrame(data_for_csv)

                    # CSV file for download
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Here is the CSV file you requested. Click below to download."
                    })
                    display_download_button(df)

                   
                    st.session_state["asked_for_csv"] = False  
                    st.rerun()

                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "No data available to generate the CSV."
                    })
                    st.rerun()

    except Exception as e:
        logger.error(f"Error in main chat interface: {e}")
        st.error(f"An error occurred: {str(e)}")
