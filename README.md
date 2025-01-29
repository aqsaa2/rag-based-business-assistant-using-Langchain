# RAG-Enhanced Chat Application

This Streamlit application allows you to have an interactive chat conversation with an AI assistant that leverages Retrieval-Augmented Generation (RAG) for improved context and relevance. By uploading documents or providing URLs, you can enhance the conversation with relevant information, leading to more informative and insightful responses.

**Features:**

*   **Context-aware chat:** Upload documents or provide URLs to establish context for your conversation.
*   **Enhanced responses:** The AI assistant utilizes RAG to understand and incorporate relevant information from uploaded documents or URLs, leading to more focused and relevant responses.
*   **Progressive learning:** As you interact with the application more, the AI assistant learns and adapts to your conversation patterns, potentially improving the quality of responses over time.
*   **Persona generation:** With sufficient interaction, the application can generate personas based on your conversation, assisting in understanding user needs.
*   **User story generation:** Based on the generated personas and context, the application can generate user stories to further define user interactions.
*   **Gherkin scenario generation:** Building upon user stories, the application can create business scenarios in Gherkin format, a behavior-driven development (BDD) framework.
*   **CSV Download:** The application provides the option to download generated personas, user stories, and Gherkin scenarios in a CSV format for further analysis or documentation purposes. The outputs are generated in CSV format, not as files.

**Live Demo:**

Experience the application live at: [https://rag-based-business-assistant-using-langchain-pdfchat.streamlit.app/](https://rag-based-business-assistant-using-langchain-pdfchat.streamlit.app/)

**Requirements:**

*   Python 3.10
*   Streamlit
*   langchain
*   langchain-google-genai (requires Google Cloud Platform project and API key)
*   chromadb
*   pandas

**Installation:**

1.  Create a virtual environment (recommended).
2.  Install the required libraries:

    ```bash
    pip install streamlit langchain langchain-google-genai chromadb pandas
    ```

3.  Obtain a Google Cloud Platform project and API key for the Google Generative AI service and store them as environment variables:

    ```bash
    export GOOGLE_API_KEY=your_api_key_here
    ```

**Usage:**

1.  Clone this repository.
2.  Run the application:

    ```bash
    streamlit run app.py
    ```

3.  Interact with the chat interface.
4.  Upload documents or provide URLs to enhance the conversation.
5.  As you interact more, the application may prompt you a comprehensive output in CSV format containing generated personas, user stories, and Gherkin scenarios. The more you interact the better the outputs are.
