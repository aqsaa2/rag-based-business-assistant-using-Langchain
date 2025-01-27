import streamlit as st
import os
import dotenv
import re
import time
import uuid
import logging
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores.chroma import Chroma
from test_app import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
    store_user_responses_to_db,
    get_stored_data,
    get_user_responses_from_db,
)
import chromadb
from chromadb import PersistentClient
import pysqlite3
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Persistent Client with path
chroma_db_path = "chroma_db"
chroma_client = PersistentClient(path=chroma_db_path)

# Load environment variables
dotenv.load_dotenv()

# Streamlit page configuration
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

if "tree_data" not in st.session_state:
    st.session_state.tree_data = {
        "Personas": [],
        "User Stories": [],
        "Business Scenarios": []
    }

# Rate-limiting mechanism
LAST_API_CALL_TIME = 0
API_CALL_DELAY = 2  # Delay in seconds between API calls

def rate_limit():
    """
    Ensures a delay between API calls to avoid hitting rate limits.
    """
    global LAST_API_CALL_TIME
    elapsed_time = time.time() - LAST_API_CALL_TIME
    if elapsed_time < API_CALL_DELAY:
        sleep_time = API_CALL_DELAY - elapsed_time
        logger.info(f"Rate limiting: Sleeping for {sleep_time:.2f} seconds.")
        time.sleep(sleep_time)
    LAST_API_CALL_TIME = time.time()

# Sidebar configuration
with st.sidebar:
    # API Key Management
    if "GOOGLE_API_KEY" not in os.environ:
        with st.expander("🔐 API Key Configuration"):
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
            ("vector_db" in st.session_state and st.session_state.vector_db is not None)
            or ("chroma_db" in st.session_state and st.session_state.chroma_db is not None)
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
            st.session_state.tree_data = {
                "Personas": [],
                "User Stories": [],
                "Business Scenarios": []
            }
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

def display_tree_with_expanders():
    """
    Displays the generated artifacts in a tree-like structure with expandable sections.
    """
    st.write("### Generated Artifacts")

    # Personas Section
    with st.expander("-- (Tree Node1) Persona: (+ Text Viewer)"):
        if st.session_state.tree_data["Personas"]:
            for persona in st.session_state.tree_data["Personas"]:
                st.write(f"--------- <{persona}>")
        else:
            st.write("--------- <No personas generated yet>")

    # User Stories Section
    with st.expander("-- (Tree Node2) User Story: (+ Text Viewer)"):
        if st.session_state.tree_data["User Stories"]:
            for user_story in st.session_state.tree_data["User Stories"]:
                st.write(f"--------- <{user_story}>")
        else:
            st.write("--------- <No user stories generated yet>")

    # Business Scenarios Section
    with st.expander("-- (Tree Node3) Business Scenario: (+ Text Viewer)"):
        if st.session_state.tree_data["Business Scenarios"]:
            for scenario in st.session_state.tree_data["Business Scenarios"]:
                st.markdown(f"```gherkin\n{scenario}\n```")
        else:
            st.write("--------- <No business scenarios generated yet>")

def generate_personas(stored_data, user_responses):
    """
    Generates personas based on stored data and user responses.
    """
    try:
        if not stored_data and not user_responses:
            return ["No data available to generate personas."]

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
        """

        # Rate limit before making the API call
        rate_limit()

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=os.environ["GOOGLE_API_KEY"],
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
                    persona_response = str(persona_response)  # Force to string as a last resort
                except Exception as e:
                    logger.error(f"Error converting the response to string: {e}")
                    return ["LLM returned an unexpected response format. Check Logs."]

        personas = [p.strip() for p in re.split(r"---PERSONA_SEPARATOR---", persona_response) if p.strip()]
        print(f"Generated Personas: {personas}")
        return personas

    except Exception as e:
        logger.error(f"Error generating personas: {e}")
        return ["Error generating personas. Check Logs."]

def generate_user_stories(persona, stored_data, user_responses):
    """
    Generates user stories based on a persona, stored data, and user responses.
    """
    try:
        if not stored_data and not user_responses:
            return ["No data available to generate user stories."]

        user_response_texts = [response["text"] for response in user_responses]
        all_text = " ".join(stored_data + user_response_texts)

        prompt = f"""Given the persona:
        {persona}

        And the following information:
        {all_text}

        Generate 2 user stories in the format: "As a [user type], I want [goal] so that [benefit]".
        """

        # Rate limit before making the API call
        rate_limit()

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            max_retries=1
        )

        user_story_response = llm([HumanMessage(content=prompt)]).content

        if not isinstance(user_story_response, str):
            if isinstance(user_story_response, AIMessage):
                user_story_response = user_story_response.content
                logger.warning(f"LLM returned AIMessage. Extracting content: {user_story_response}")
            else:
                try:
                    user_story_response = json.dumps(user_story_response)
                    logger.warning(f"LLM returned non-string response. Attempting JSON Conversion: {user_story_response}")
                except TypeError as e:
                    logger.error(f"LLM returned unexpected non-string, non-JSON and non AIMessage response: {type(user_story_response)}, {user_story_response}, {e}")
                    user_story_response = str(user_story_response)  # Force to string as a last resort
                except Exception as e:
                    logger.error(f"Error converting the response to string: {e}")
                    return ["LLM returned an unexpected response format. Check Logs."]

        user_stories = [s.strip() for s in user_story_response.split("\n") if s.strip()]
        print(f"Generated User Stories: {user_stories}")
        return user_stories

    except Exception as e:
        logger.error(f"Error generating user stories: {e}")
        return ["Error generating user stories. Check Logs."]

def generate_gherkin_scenarios(user_story, stored_data, user_responses):
    """
    Generates Gherkin scenarios based on a user story, stored data, and user responses.
    """
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
        Format each scenario like this:
        ---SCENARIO_SEPARATOR---
        Scenario: [Scenario Name]
        Given [precondition]
        When [action]
        Then [expected result]
        ---SCENARIO_SEPARATOR---
        """

        # Rate limit before making the API call
        rate_limit()

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=os.environ["GOOGLE_API_KEY"],
            max_retries=1
        )

        gherkin_response = llm([HumanMessage(content=prompt)]).content

        if not isinstance(gherkin_response, str):
            if isinstance(gherkin_response, AIMessage):
                gherkin_response = gherkin_response.content
                logger.warning(f"LLM returned AIMessage. Extracting content: {gherkin_response}")
            else:
                try:
                    gherkin_response = json.dumps(gherkin_response)
                    logger.warning(f"LLM returned non-string response. Attempting JSON Conversion: {gherkin_response}")
                except TypeError as e:
                    logger.error(f"LLM returned unexpected non-string, non-JSON and non AIMessage response: {type(gherkin_response)}, {gherkin_response}, {e}")
                    gherkin_response = str(gherkin_response)  # Force to string as a last resort
                except Exception as e:
                    logger.error(f"Error converting the response to string: {e}")
                    return ["LLM returned an unexpected response format. Check Logs."]

        scenarios = [s.strip() for s in re.split(r"---SCENARIO_SEPARATOR---", gherkin_response) if s.strip()]
        print(f"Generated Gherkin Scenarios: {scenarios}")
        return scenarios

    except Exception as e:
        logger.error(f"Error generating Gherkin scenarios: {e}")
        return ["Error generating Gherkin scenarios. Check Logs."]

def save_and_display_as_csv(personas, user_stories, gherkin_scenarios):
    """Saves generated data to a CSV and displays it using Streamlit."""
    try:
        data = []
        for i, persona in enumerate(personas):
            persona_data = {"Type": "Persona", "Content": persona}
            data.append(persona_data)

        for i, user_story in enumerate(user_stories):
            story_data = {"Type": "User Story", "Content": user_story}
            data.append(story_data)

        for i, scenario in enumerate(gherkin_scenarios):
            scenario_data = {"Type": "Gherkin Scenario", "Content": scenario}
            data.append(scenario_data)

        df = pd.DataFrame(data)
        st.write("### Generated Data (CSV Format)")
        st.dataframe(df)  # Display as a DataFrame
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv_data,
            file_name='generated_artifacts.csv',
            mime='text/csv',
        )
        return df
    except Exception as e:
        logger.error(f"Error saving/displaying CSV: {e}")
        st.error(f"Error creating CSV: {e}")
        return None



if not google_api_key and "GOOGLE_API_KEY" not in os.environ:
    st.warning("⚠️ Please configure your API key in the sidebar to continue")
else:
    try:
        llm_stream = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.4,
            streaming=True,
            google_api_key=google_api_key,
            max_retries=1
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

                # Check to see if enough user input has been gathered to generate artifacts
                if st.session_state.user_input_count >= 5:
                    stored_data = get_stored_data() 
                    user_responses = get_user_responses_from_db(st.session_state.session_id) 

                    if stored_data:
                        # Generate Personas
                        try:
                            personas = generate_personas(stored_data, user_responses)
                            st.session_state.tree_data["Personas"] = personas
                        except Exception as e:
                            logger.error(f"Error generating personas: {e}")
                            st.session_state.tree_data["Personas"] = ["Error generating personas. Check Logs."]

                        # Generate User Stories
                        try:
                            user_stories = []
                            for persona in personas:
                                stories = generate_user_stories(persona, stored_data, user_responses)
                                user_stories.extend(stories)
                            st.session_state.tree_data["User Stories"] = user_stories
                        except Exception as e:
                            logger.error(f"Error generating user stories: {e}")
                            st.session_state.tree_data["User Stories"] = ["Error generating user stories. Check Logs."]

                        # Generate Gherkin Scenarios
                        try:
                            gherkin_scenarios = []
                            for user_story in user_stories:
                                scenarios = generate_gherkin_scenarios(user_story, stored_data, user_responses)
                                gherkin_scenarios.extend(scenarios)
                            st.session_state.tree_data["Business Scenarios"] = gherkin_scenarios
                        except Exception as e:
                            logger.error(f"Error generating Gherkin scenarios: {e}")
                            st.session_state.tree_data["Business Scenarios"] = ["Error generating Gherkin scenarios. Check Logs."]

                        # Display the updated tree with expanders
                        df = save_and_display_as_csv(personas, user_stories, gherkin_scenarios)
                        # display_tree_with_expanders()

    except Exception as e:
        logger.error(f"Error in main chat interface: {e}")
        st.error(f"An error occurred: {str(e)}")
