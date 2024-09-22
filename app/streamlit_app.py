import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
import tempfile
import re
from io import StringIO
import tiktoken


st.title("🧾 Extraction d'informations sur documents public")

openai_api_key = st.sidebar.text_input("Clé d'API OpenAI", type="password")

uploaded_file = st.sidebar.file_uploader("Dépose ton document", type=['pdf'])

# Function to count tokens
def count_tokens(text):
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(encoding.encode(text))

# Function to estimate price (using GPT-3.5-turbo pricing as an example)
def estimate_price(token_count):
    return (token_count / 1000) * 0.002  # $0.002 per 1k tokens


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
    doc_token_count = count_tokens(full_text)
    doc_price_estimate = estimate_price(doc_token_count)
    side_container = st.sidebar.container(border=True)
    side_container.write(f"🧮 Nombre de tokens dans le document: {doc_token_count}")
    side_container.write(f"💵 Estimation du prix de traitement de ce document: ${doc_price_estimate:.6f}")
    


def generate_response(input_text):
    model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
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

1) Qui est l'auteur du document
2) Quel est le titre du document

Pour chaque question, répond en mentionnant l'extrait du texte qui te permet de répondre ainsi que la page où se trouve l'extrait.

Format de la réponse:

> {numero question}. {réponse}
   - Extrait du texte: <<"{extrait}">>
   - Page: {page}

Exemple de réponse: 

> 1. La devise est l'euro
   - Extrait du texte: <<"Devise : EUR">>
   - Page: 3 sur 4""",height=350
    )
    if not openai_api_key.startswith("sk-"):
        st.warning("S'il te plait spécifie ta clé API", icon="⚠")
    elif uploaded_file is None:
        st.warning("S'il te plait dépose ton document", icon="⚠")
        
        
    # Count tokens and estimate price for the query
    query_token_count = count_tokens(text)
    query_price_estimate = estimate_price(query_token_count)
    
    total_tokens = 0
    if uploaded_file is not None:
        # Total tokens and price (document + query)
        total_tokens = doc_token_count + query_token_count
        total_price = doc_price_estimate + query_price_estimate
        
        container = st.container(border=True)
        container.write(f"🧮 Nombre de tokens total: {total_tokens}")
        container.write(f"💵 Estimation du prix de la requête: ${total_price:.6f}")
        
    if total_tokens > 200000:
        st.warning("Attention, la limite de maximum de tokens est dépassé, change le document ou réduis le nombre de question. Limite: 200 000", icon="⚠")
    
    submitted = st.form_submit_button("Envoyer la requête")
    if submitted and openai_api_key.startswith("sk-") and uploaded_file is not None and total_tokens <= 200000:
        generate_response(text)
