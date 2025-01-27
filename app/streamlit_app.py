import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain_core.documents import Document
import tempfile
import re
from io import StringIO
import tiktoken
from openai import OpenAI
from mistralai import Mistral
import anthropic
import textwrap


def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

def estimate_price(token_count):
    return (token_count / 1000000) * prix

def is_openAI():
    return selected_distrib == "OpenAI"

def is_openAI_and_key_set():
    return is_openAI() and openai_api_key is not None and openai_api_key != ''

def is_anthropic():
    return selected_distrib == "Anthropic"

def is_anthropic_and_key_set():
    return is_anthropic() and anthropic_api_key is not None and anthropic_api_key != ''


def is_mistral():
    return selected_distrib == "Mistral"

def is_mistral_and_key_set():
    return is_mistral() and mistral_api_key is not None and mistral_api_key != ''

def get_models_list():
    models_id = []
    if is_openAI_and_key_set():
        client = OpenAI(api_key=openai_api_key)
        models = client.models.list()
        for model in models:
            model_id = model.id
            if ("gpt" in model_id or "o1" in model_id) and "preview" not in model_id:
                models_id.append(model_id)
    elif is_anthropic_and_key_set():
        client = anthropic.Anthropic(api_key=anthropic_api_key)
        models = client.models.list()
        for model in models:
            model_id = model.id
            #if "gpt" in model_id and "preview" not in model_id:
            models_id.append(model_id)
    elif is_mistral_and_key_set():
        client = Mistral(api_key=mistral_api_key)
        models = client.models.list().data
        for model in models:
            model_id = model.id
            #if "gpt" in model_id and "preview" not in model_id:
            models_id.append(model_id)
    return models_id
    


model_infos = {
                "gpt-4o": {"price" : 2.5, "context" : 128000},
                "gpt-4o-mini": {"price" : 0.150, "context" : 128000},
                "o1": {"price" : 15, "context" : 200000},
                "o1-mini": {"price" : 3, "context" : 200000},
                "gpt-3.5-turbo": {"price" : 3, "context" : 16498},
                "claude-3-5-sonnet-20241022": {"price" : 3, "context" : 200000},
                "claude-3-5-haiku-20241022": {"price" : 0.8, "context" : 200000},
                "claude-3-opus-20240229": {"price" : 15, "context" : 200000},
                "mistral-large-latest": {"price" : 2, "context" : 128000},
               }

st.title("🧾 Extraction d'informations sur documents public")

selected_distrib = st.sidebar.selectbox("Sélection du distributeur", options=("OpenAI", "Anthropic", "Mistral"), placeholder="Selectionne un distributeur...")

openai_api_key = None
if is_openAI():
    openai_api_key = st.sidebar.text_input("Clé d'API OpenAI", type="password")

anthropic_api_key = None
if is_anthropic():
    anthropic_api_key = st.sidebar.text_input("Clé d'API Anthropic", type="password")
    
mistral_api_key = None
if is_mistral():
    mistral_api_key = st.sidebar.text_input("Clé d'API Mistral", type="password")

uploaded_file = st.sidebar.file_uploader("Dépose ton document", type=['pdf'])

info = st.sidebar.text_input("Information recherchée", type="default")

selected_model = None
if is_openAI_and_key_set() or is_anthropic_and_key_set() or is_mistral_and_key_set():
    selected_model = st.sidebar.selectbox("Sélection du modèle", options=get_models_list(), placeholder="Selectionne un modèle...")

prix = None
context = None
if selected_model is not None:
    prix = -1
    context = -1
    if selected_model in model_infos: 
        prix = model_infos[selected_model]["price"]
        context= model_infos[selected_model]['context']
        
    container = st.sidebar.container(border=True)
    container.write("Information modèle:")
    container.write(f"💲Prix: {prix}$/M Input tokens")
    container.write(f"📖 Contexte Max {context} tokens")


if uploaded_file is not None:
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    #Load the PDF
    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    
    # Extract text content from documents
    text_content = []
    for doc in documents:
        text_content.append(doc.page_content.lower())
    full_text = "".join(text_content)
    
     # Display token count and estimated price for the uploaded document
    if prix is not None:
        doc_token_count = count_tokens(full_text)
        doc_price_estimate = estimate_price(doc_token_count)
        side_container = st.sidebar.container(border=True)
        side_container.write(f"🧮 Nombre de tokens dans le document: {doc_token_count}")
        side_container.write(f"💵 Estimation du prix de traitement de ce document: ${doc_price_estimate:.6f}")
    


def generate_response(input_text):
    input_text = input_text.replace("{info}", info)
    model = None
    if is_openAI():
         model = ChatOpenAI(temperature=0.7, api_key=openai_api_key, model=selected_model)
    elif is_anthropic():
         model = ChatAnthropic(temperature=0.7, anthropic_api_key=anthropic_api_key, model=selected_model)
    elif is_mistral():
         model = ChatMistralAI(temperature=0.7, api_key=mistral_api_key, model=selected_model)
    if total_tokens > context:
        batchs_response = []
        batch_amount = int(total_tokens/context)+2
        st.write(batch_amount)
        batch_documents = []
        documents_per_batch = len(documents) // batch_amount

        # Handle remaining documents
        remaining = len(documents) % batch_amount

        for i in range(batch_amount):
            start_idx = i * documents_per_batch + min(i, remaining)
            end_idx = start_idx + documents_per_batch + (1 if i < remaining else 0)
            batch = documents[start_idx:end_idx]
            batch_documents.append(batch)
            
        i = 0
        for batch in batch_documents:
            chain = load_qa_chain(model,verbose=True)
            response = chain.run(input_documents=batch, question=input_text)
            response = ">>> NOUVEAU BATCH "+str(i)+" >>>"+response
            st.write(response)
            batchs_response.append(response)
            i=i+1
        # Resume
        resume_prompt = """Voici une séquence de plusieurs batch de réponse, le numéro des réponses est le même. Ton objectif est de sélectionner un réponse parmis toutes celle dans les batch, si possible la plus cohérente. Si une réponse est remplie sur un des bacth prioritise là dans ton résumé. Tu dois garder le même format > {numero question}. {question}
   - Réponse: {réponse}
   - Extrait du texte: <<"{extrait}">>
   - Page: {page}\n"""

        docs = [
        Document(
            page_content=text,
            metadata={"index": i}  # Optional metadata
        ) for i, text in enumerate(batchs_response)
]
        chain = load_qa_chain(model,verbose=True)
        final_response = chain.run(input_documents=docs, question=resume_prompt)
        verify_response(final_response)
    else:
        chain = load_qa_chain(model,verbose=True)
        response = chain.run(input_documents=documents, question=input_text)
        verify_response(response)
        #st.info(model.invoke(input_text))
    
def extract_text_between_markers(text):
    pattern = r'<<"\s*(.*?)\s*">>'
    matches = re.findall(pattern, text)
    return matches
    
def verify_response(response):
    extracted_text = extract_text_between_markers(response)
    for t in extracted_text:
        if normalize_spaces(t.lower()) in normalize_spaces(full_text):
            response = response.replace("<<\""+t+"\">>", "\""+t+"\"✅(Valide: L'extrait est dans le document)")
        else:
            response = response.replace("<<\""+t+"\">>", "\""+t+"\" (Invalide: L'extrait n'apparait pas dans le document, le modèle d'IA a peut être inventé cette réponse)")
    st.info(response)
    
def normalize_spaces(s):
    return " ".join(s.split())

with st.form("my_form"):
    text = st.text_area(
        "Entre tes questions:",
        """Questions:

1) Qui est l'auteur du document {info} ?
2) Quel est le titre du document  {info} ?

Pour chaque question, répond en mentionnant l'extrait du texte qui te permet de répondre ainsi que la page où se trouve l'extrait.

Format de la réponse:

> {numero question}. {question}
   - Réponse: {réponse}
   - Extrait du texte: <<"{extrait}">>
   - Page: {page}

Exemple de réponse: 

> 1. Qui est l'auteur du document ?
   - Réponse: L'auteur est Marcel Drick
   - Extrait du texte: <<"Document écrit par Marcel Drick">>
   - Page: 3 sur 4""",height=350
    )
    if not(is_anthropic_and_key_set()) and not(is_openAI_and_key_set()) and not(is_mistral_and_key_set()):
        st.warning("S'il te plait spécifie ta clé API", icon="⚠")
    elif uploaded_file is None:
        st.warning("S'il te plait dépose ton document", icon="⚠")
    elif selected_model is None:
        st.warning("S'il te plait sélectionne un modèle", icon="⚠")
    elif selected_model not in model_infos:
        st.warning("Modèle non supporté", icon="⛔")
        
        
    
    total_tokens = 0
    if prix is not None and uploaded_file is not None and selected_model is not None and selected_model in model_infos:
        # Count tokens and estimate price for the query
        query_token_count = count_tokens(text)
        query_price_estimate = estimate_price(query_token_count)
        # Total tokens and price (document + query)
        total_tokens = doc_token_count + query_token_count
        total_price = doc_price_estimate + query_price_estimate
        
        container = st.container(border=True)
        container.write(f"🧮 Nombre de tokens total: {total_tokens}")
        container.write(f"💵 Estimation du prix de la requête: ${total_price:.6f}")
        
    if context is not None and total_tokens > context:
        st.warning("Attention, la limite de maximum de tokens est dépassé, l'analyse va se faire en plusieurs batchs puis faire un résumé", icon="⚠")
    
    submitted = st.form_submit_button("Envoyer la requête")
    if submitted and selected_model is not None and selected_model in model_infos and uploaded_file is not None:
        generate_response(text)
