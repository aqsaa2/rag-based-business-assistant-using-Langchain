import streamlit as st
import os
import dotenv
import uuid
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores.chroma import Chroma
from test import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
    store_user_responses_to_db,
    # display_all_chroma_data,
    # generate_outputs,
    # display_stored_data,
)
from chromadb import PersistentClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

chroma_db_path = "chroma_db"
chroma_client = PersistentClient(path=chroma_db_path)

# # Reset database (delete existing collections)
# try:
#     chroma_client.delete_collection(name="user_responses")
#     logger.info("Chroma database reset successfully.")
# except Exception as e:
#     logger.warning(f"Failed to reset Chroma database: {e}")

# Load environment variables
dotenv.load_dotenv()

# # Initialize Chroma DB client
# collection = chroma_client.get_or_create_collection(name="user_responses")

# def display_stored_data():
#     try:
#         # Fetch all data from the Chroma collection
#         results = collection.get()
#         if results:
#             st.subheader("Stored Data in Database:")
#             for idx, doc in enumerate(results["documents"]):
#                 metadata = results["metadatas"][idx]
#                 st.write(f"**Document {idx + 1}:** {doc}")
#                 st.write(f"**Metadata:** {metadata}")
#                 st.divider()
#         else:
#             st.write("No data stored in the database yet.")
#     except Exception as e:
#         logger.error(f"Failed to fetch data: {e}")
#         st.error(f"Error fetching data: {str(e)}")

# Page configuration
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
            ("vec_db" in st.session_state and st.session_state.vector_db is not None) or
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

# Initialize session state for tracking context-building
if "user_input_count" not in st.session_state:
    st.session_state.user_input_count = 0

if "ask_for_csv" not in st.session_state:
    st.session_state.ask_for_csv = False

# Main chat interface
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

        # Chat input for answering the business questions
        if prompt := st.chat_input("Type your response here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            store_user_responses_to_db(prompt, st.session_state.session_id)  # Save user response to Chroma DB
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

                if not st.session_state.use_rag:
                    st.write_stream(stream_llm_response(llm_stream, messages))
                else:
                    st.write_stream(stream_llm_rag_response(llm_stream, messages))

            # Check if enough user input has been gathered to generate persona
            # Track the number of responses
            if st.session_state.user_input_count >= 4 and "asked_for_csv" not in st.session_state:
                st.session_state["asked_for_csv"] = True
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I have gathered enough information. Would you like me to generate a CSV file with personas, user stories, and business scenarios? (Type 'yes' to confirm)"
                })
                st.rerun()

        # Handle persona generation confirmation
        if "asked_for_csv" in st.session_state and st.session_state["asked_for_csv"]:
            last_message = st.session_state.messages[-1]["content"]
            if last_message.strip().lower() == "yes":
                # Call function to generate the CSV
                stored_data = get_stored_data()
                if stored_data:
                    csv_data = generate_csv_from_chroma(llm_stream, stored_data)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Here is the CSV file you requested. Click below to download."
                    })
                    # download button for the user
                    display_download_button(llm_stream, stored_data)
                    st.session_state["asked_for_csv"] = False  
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "No data available to generate the CSV."
                    })
                st.rerun()

    except Exception as e:
        logger.error(f"Error in main chat interface: {e}")
        st.error(f"An error occurred: {str(e)}")