# pip install streamlit langchain langchain-huggingface beautifulsoup4 python-dotenv chromadb requests

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import LLMResult, Generation
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import requests
import os
from typing import Any, List, Optional
from pydantic import Field

load_dotenv()

# Configuration  du modèle
DEFAULT_MODEL = "deepseek/deepseek-r1-0528-qwen3-8b:free"

# Classe personnalisée pour OpenRouter
class OpenRouterLLM(BaseLLM):
    api_key: str = Field(..., description="OpenRouter API key")
    model: str = Field(default=DEFAULT_MODEL, description="Model to use")
    base_url: str = Field(default="https://openrouter.ai/api/v1/chat/completions", description="OpenRouter API URL")
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            return f"Erreur lors de l'appel à OpenRouter: {str(e)}"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Generate responses for multiple prompts."""
        generations = []
        for prompt in prompts:
            response = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=response)])
        
        return LLMResult(generations=generations)

def create_llm():
    """Fonction utilitaire pour créer une instance LLM"""
    return OpenRouterLLM(api_key=os.getenv("OPENROUTER_API_KEY"))

def get_vectorstore_from_url(url):
    """Crée un vectorstore à partir d'une URL en utilisant HuggingFace embeddings"""
    
    # Chargement de document
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # Division en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    document_chunks = text_splitter.split_documents(document)
    
    # Utilisation de HuggingFace embeddings 
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # CPU pour éviter les problèmes de GPU
    )
    
    # Création de vectorstore
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    return vector_store

def get_context_retriever_chain(vector_store):
    """Crée une chaîne de récupération contextuelle"""
    
    # model
    llm = create_llm()
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

def get_conversational_rag_chain(retriever_chain): 
    """Crée la chaîne RAG conversationnelle"""
    
    # model
    llm = create_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the user's questions ONLY based on the below context. 
        If the context doesn't contain relevant information for the question, 
        clearly state that you cannot help with topics outside the provided documentation.
        
        Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    """Génère une réponse basée sur l'input utilisateur"""
    try:
        retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
        conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
        
        response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": user_input
        })
        
        return response['answer']
    except Exception as e:
        return f"Erreur lors de la génération de la réponse: {str(e)}"

# Configuration de l'app
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites 🤖")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL", placeholder="https://example.com")
    

    st.markdown("---")
    
    # Vérification des clés API
    if not os.getenv("OPENROUTER_API_KEY"):
        st.error("Clé API OpenRouter manquante!")
        st.markdown("Ajoutez OPENROUTER_API_KEY dans votre fichier .env")

if website_url is None or website_url == "":
    st.info(" Please enter a website URL in the sidebar")
else:
    # State management
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm ready to help you with questions about the website content. How can I assist you?"),
        ]
    
    if "vector_store" not in st.session_state:
        with st.spinner("Loading and processing website content..."):
            try:
                st.session_state.vector_store = get_vectorstore_from_url(website_url)
                st.success("Website content loaded successfully!")
            except Exception as e:
                st.error(f"Error loading website: {str(e)}")
                st.stop()
    
    # User input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
    
    # Display conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI", avatar="🤖"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human", avatar="👤"):
                st.write(message.content)