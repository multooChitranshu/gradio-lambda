import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import boto3
from botocore.exceptions import ClientError
from langchain_community.graphs import Neo4jGraph
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# RAG Integration imports
from pinecone import Pinecone
from langchain_pinecone import Pinecone as LangchainPineconeVectorstore
from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws.chat_models import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
import re
from langchain_core.documents import Document
from typing import Union, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}

os.environ['PINECONE_API_KEY'] = 'pcsk_2hDYAK_FgFgg7oKf75z7o5XdSYfWSyFaUhysAFg32S9M57iX5YhMwGxgXgN9Phd5tK8eAc'
os.environ['PINECONE_ENVIRONMENT'] = 'us-east-1'
os.environ['INDEX_NAME'] = 'smart-saving-unstruct'
os.environ['PINECONE_INDEX_HOST'] = 'https://smart-saving-unstruct-1vfchtc.svc.aped-4627-b74a.pinecone.io'
os.environ['AWS_REGION_1'] = 'us-east-1'
os.environ['EMBEDDING_MODEL_ID'] = 'amazon.titan-embed-text-v2:0'
os.environ['GENERATION_MODEL_ID'] = 'anthropic.claude-3-sonnet-20240229-v1:0'
os.environ['NEO4J_URI'] = 'neo4j+s://3af8a684.databases.neo4j.io'
os.environ['NEO4J_USERNAME'] = 'neo4j'
os.environ['NEO4J_PASSWORD'] = 'Tat-pgntlgkxbtIyxdNf5W1P0qorkDBm3YwhRWzrY4k'


# --- RAG Configuration (from Environment Variables) ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("INDEX_NAME")
PINECONE_INDEX_HOST = os.getenv("PINECONE_INDEX_HOST")
AWS_REGION_1 = os.getenv("AWS_REGION_1", "us-east-1")
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
GENERATION_MODEL_ID = os.getenv("GENERATION_MODEL_ID")

# Neo4j Configuration for RAG
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://3af8a684.databases.neo4j.io")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Tat-pgntlgkxbtIyxdNf5W1P0qorkDBm3YwhRWzrY4k")


pc_client = None
bedrock_runtime_client = None
savings_rag_chain = None
risk_rag_chain=None
embeddings_instance = None
vectorstore_instance = None
llm_instance = None
neo4j_driver = None
classification_chain = None

def fetch_client_data():
    """Fetch client data with error handling using new Cypher query"""
    try:
        NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://3af8a684.databases.neo4j.io")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Tat-pgntlgkxbtIyxdNf5W1P0qorkDBm3YwhRWzrY4k")
        
        kg = Neo4jGraph(
            url=NEO4J_URI, 
            username=NEO4J_USER, 
            password=NEO4J_PASSWORD,
            timeout=10
        )
        
        cypher = """
        MATCH (c:Customer)
        OPTIONAL MATCH (c)-[:HAS_CREDIT_REPORT]->(cr:CreditReport)
        OPTIONAL MATCH (c)-[:HAS_FINANCIAL_GOAL]->(f:FinancialGoal)
        OPTIONAL MATCH (c)-[:OWES_DEBT]->(d:Debt)
        RETURN 
        c.id AS id,
        c.name AS name,
        c.age AS age,
        c.monthly_income_total AS income,
        cr.score AS credit_score,
        SUM(f.currentSaving) AS total_savings,
        SUM(d.remaining_balance) AS total_debt,
        c.employerIndustry AS industry,
        c.state as city,
        f.description as savings_goal
        """
        
        results = kg.query(cypher)
        client_data = {}
        for record in results:
            name = record["name"]
            # Store both name and id for each client with updated structure
            client_data[name] = {
                "id": record["id"],
                "income": record["income"] or 0,  # Handle null values
                "monthly_expenses": record["income"] * 0.6 if record["income"] else 0,  # Estimate since not in query
                "savings": record["total_savings"] or 0,
                "credit_score": record["credit_score"] or 650,  # Default if missing
                "occupation": record["industry"] or "Unknown",  # Changed from occupation
                "age": record["age"] or 30,
                "debt": record["total_debt"] or 0,
                "city": record["city"] or "Unknown",
                "savings_goal": record["savings_goal"] or "General Savings"
            }
        
        return client_data, None
        
    except Exception as e:
        logger.info(f"ERROR: Failed to connect to Neo4j: {e}")
        # Return enhanced mock data with new structure
        mock_data = {
            "Sarah Johnson": {
                "id": "P001",
                "income": 75000,
                "monthly_expenses": 4500,
                "savings": 25000,
                "credit_score": 750,
                "occupation": "Technology",
                "age": 32,
                "debt": 15000,
                "city": "Mumbai",
                "savings_goal": "Home Purchase Fund"
            },
            "John Doe": {
                "id": "P002",
                "income": 65000,
                "monthly_expenses": 3800,
                "savings": 18000,
                "credit_score": 720,
                "occupation": "Finance",
                "age": 28,
                "debt": 12000,
                "city": "Delhi",
                "savings_goal": "Emergency Fund"
            },
            "Michael Chen": {
                "id": "P003",
                "income": 90000,
                "monthly_expenses": 5200,
                "savings": 45000,
                "credit_score": 780,
                "occupation": "Healthcare",
                "age": 35,
                "debt": 8000,
                "city": "Bangalore",
                "savings_goal": "Retirement Planning"
            },
            "Priya Sharma": {
                "id": "P004",
                "income": 55000,
                "monthly_expenses": 3200,
                "savings": 12000,
                "credit_score": 690,
                "occupation": "Education",
                "age": 29,
                "debt": 18000,
                "city": "Pune",
                "savings_goal": "Child Education Fund"
            }
        }
        return mock_data, str(e)

CLIENT_DATA, error = fetch_client_data()

def initialize_classifier():
    """Initialize the classification components."""
    global bedrock_runtime_client, classification_chain, llm_instance
    
    print("Initializing query classifier...")
    
    # Initialize AWS Bedrock Runtime Client
    if bedrock_runtime_client is None:
        bedrock_runtime_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=AWS_REGION_1
        )
        print("✓ AWS Bedrock client initialized")
    
    # Initialize LLM
    if llm_instance is None:
        claude_model_kwargs = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "temperature": 0.1,
            "top_p": 0.9
        }
        llm_instance = ChatBedrock(
            model_id=GENERATION_MODEL_ID,
            client=bedrock_runtime_client,
            model_kwargs=claude_model_kwargs
        )
        print("✓ LLM initialized")
    
    # Build classification chain
    if classification_chain is None:
        prompt_template = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a query classifier for financial services. 
                Classify the user query into exactly one category:
                - "risk" for queries about: credit risk, loan eligibility, risk assessment, default probability, creditworthiness, loan approval
                - "savings" for queries about: savings analysis, debt-to-income ratio, financial health, savings recommendations, budgeting, spending analysis
                
                Respond with only one word: "risk" or "savings" (lowercase, no quotes).
                """
            ),
            ("user", "Classify this query: {query}")
        ])
        
        classification_chain = (
            prompt_template
            | llm_instance
            | StrOutputParser()
        )
        print("✓ Classification chain initialized")

def initialize_savings_rag_components():
    """
    Initializes all necessary RAG clients and LangChain components.
    This function should be called only once per Lambda container lifecycle.
    """
    global pc_client, bedrock_runtime_client, savings_rag_chain, embeddings_instance, vectorstore_instance, llm_instance, neo4j_driver

    # --- Initialize Pinecone Client ---
    if pc_client is None:
        if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
            raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT not set as Lambda environment variables.")
        try:
            pc_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            print("Successfully initialized Pinecone client.")
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

    # --- Initialize AWS Bedrock Runtime Client ---
    if bedrock_runtime_client is None:
        if not AWS_REGION_1:
            raise ValueError("AWS_REGION_1 not set as a Lambda environment variable.")
        try:
            bedrock_runtime_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=AWS_REGION_1
            )
            print(f"Successfully initialized AWS Bedrock client in region {AWS_REGION_1}.")
        except Exception as e:
            print(f"Error initializing AWS Bedrock client: {e}")
            raise

    # --- Initialize LangChain Embeddings ---
    if embeddings_instance is None:
        print(f"Initializing embedding model {EMBEDDING_MODEL_ID} for LangChain...")
        embeddings_instance = BedrockEmbeddings(
            model_id=EMBEDDING_MODEL_ID,
            client=bedrock_runtime_client
        )

    # --- Initialize LangChain Pinecone Vectorstore ---
    if vectorstore_instance is None:
        if not INDEX_NAME:
            raise ValueError("INDEX_NAME not set as a Lambda environment variable.")
        print(f"Initializing LangChain Pinecone Vectorstore using index '{INDEX_NAME}'...")
        try:
            vectorstore_instance = LangchainPineconeVectorstore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=embeddings_instance,
                text_key="original_content"
            )
            print("LangChain Pinecone vector store initialized from existing index.")
        except Exception as e:
            print(f"Error initializing LangChain Pinecone Vectorstore: {e}")
            raise

    # --- Initialize Neo4j Driver ---
    if neo4j_driver is None:
        if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
            print("Neo4j credentials not fully set. Skipping Neo4j driver initialization.")
        else:
            try:
                neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
                neo4j_driver.verify_connectivity()
                print("Successfully initialized Neo4j driver.")
            except ServiceUnavailable as e:
                print(f"Neo4j Service Unavailable: {e}. Check Neo4j instance or URI.")
                neo4j_driver = None
            except Exception as e:
                print(f"Error initializing Neo4j driver: {e}")
                neo4j_driver = None

    # --- Initialize LLM for generation ---
    if llm_instance is None:
        if not GENERATION_MODEL_ID:
            raise ValueError("GENERATION_MODEL_ID not set as an environment variable.")

        print(f"Initializing generation model {GENERATION_MODEL_ID} for LangChain...")
        claude_model_kwargs = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": 1,
            "top_p": 0.999
        }
        llm_instance = ChatBedrock(
            model_id=GENERATION_MODEL_ID,
            client=bedrock_runtime_client,
            model_kwargs=claude_model_kwargs
        )

    # --- Build the RAG chain ---
    if savings_rag_chain is None:
        retriever = vectorstore_instance.as_retriever(search_kwargs={"k": 3})
        print("Retriever initialized with top-k search set to 3.")

        def format_docs(docs: list[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful SAVINGS analyst for JPMorgan Chase who analyzes client's profile from financial lens.
                    Based on the following context and the detailed user profile information (if provided),
                    please answer the question accurately and concisely. Analyze the savings aspect of the question.

                    Your response should be formatted as JSON with the following structure:
                    {{
                        "type": "SAVINGS",
                        "results": {{
                            "debt_to_income_ratio": "Provide the debt-to-income ratio.(Just provide the percentage value)",
                            "financial_health": "Assess the financial health.(Based on financial health choose from Poor,Fair,Good,Excellent)",
                            "analysis_title": "Provide a title for the analysis.",
                            "recommendation": "Offer a recommendation based on the analysis.",
                            "key_factors": "List key factors affecting or improving savings."
                        }},
                        "evidence": {{
                            "knowledge_graph_reasoning": ["Provide top 3 reasoning based on the knowledge graph"],
                            "supporting_documents": ["List supporting top 3 documents"]
                        }}
                    }}
                    No matter, what your response should only be in the above json format, no additional text. If the answer is not available in the provided information, return an empty json.
                    Pay close attention to any user-specific identifiers (like user ID) and any 'Unstructured Data'
                    notes in the user profile, as these often contain critical insights.

                    Summarize her profile information and any relevant context to provide a comprehensive answer.
                    Do not make up information. Focus on providing relevant details from the context and user profile.
            
                    User Profile from Knowledge Graph:
                    {user_profile_info}
                    """
                ),
                ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
            ]
        )
        print("Prompt template initialized.")
        savings_rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: retriever.invoke(x["question"])
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(x["context"])
            )
            | prompt_template
            | llm_instance
            | StrOutputParser()
        )
        print("RAG chain initialized.")


def initialize_risk_rag_components():
    """
    Initializes all necessary RAG clients and LangChain components.
    This function should be called only once per Lambda container lifecycle.
    """
    global pc_client, bedrock_runtime_client, risk_rag_chain, embeddings_instance, vectorstore_instance, llm_instance, neo4j_driver

    # --- Initialize Pinecone Client ---
    if pc_client is None:
        if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
            raise ValueError("PINECONE_API_KEY or PINECONE_ENVIRONMENT not set as Lambda environment variables.")
        try:
            pc_client = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            print("Successfully initialized Pinecone client.")
        except Exception as e:
            print(f"Error initializing Pinecone: {e}")
            raise

    # --- Initialize AWS Bedrock Runtime Client ---
    if bedrock_runtime_client is None:
        if not AWS_REGION_1:
            raise ValueError("AWS_REGION_1 not set as a Lambda environment variable.")
        try:
            bedrock_runtime_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=AWS_REGION_1
            )
            print(f"Successfully initialized AWS Bedrock client in region {AWS_REGION_1}.")
        except Exception as e:
            print(f"Error initializing AWS Bedrock client: {e}")
            raise

    # --- Initialize LangChain Embeddings ---
    if embeddings_instance is None:
        print(f"Initializing embedding model {EMBEDDING_MODEL_ID} for LangChain...")
        embeddings_instance = BedrockEmbeddings(
            model_id=EMBEDDING_MODEL_ID,
            client=bedrock_runtime_client
        )

    # --- Initialize LangChain Pinecone Vectorstore ---
    if vectorstore_instance is None:
        if not INDEX_NAME:
            raise ValueError("INDEX_NAME not set as a Lambda environment variable.")
        print(f"Initializing LangChain Pinecone Vectorstore using index '{INDEX_NAME}'...")
        try:
            vectorstore_instance = LangchainPineconeVectorstore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=embeddings_instance,
                text_key="original_content"
            )
            print("LangChain Pinecone vector store initialized from existing index.")
        except Exception as e:
            print(f"Error initializing LangChain Pinecone Vectorstore: {e}")
            raise

    # --- Initialize Neo4j Driver ---
    if neo4j_driver is None:
        if not NEO4J_URI or not NEO4J_USERNAME or not NEO4J_PASSWORD:
            print("Neo4j credentials not fully set. Skipping Neo4j driver initialization.")
        else:
            try:
                neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
                neo4j_driver.verify_connectivity()
                print("Successfully initialized Neo4j driver.")
            except ServiceUnavailable as e:
                print(f"Neo4j Service Unavailable: {e}. Check Neo4j instance or URI.")
                neo4j_driver = None
            except Exception as e:
                print(f"Error initializing Neo4j driver: {e}")
                neo4j_driver = None

    # --- Initialize LLM for generation ---
    if llm_instance is None:
        if not GENERATION_MODEL_ID:
            raise ValueError("GENERATION_MODEL_ID not set as an environment variable.")

        print(f"Initializing generation model {GENERATION_MODEL_ID} for LangChain...")
        claude_model_kwargs = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "top_k": 250,
            "stop_sequences": [],
            "temperature": 1,
            "top_p": 0.999
        }
        llm_instance = ChatBedrock(
            model_id=GENERATION_MODEL_ID,
            client=bedrock_runtime_client,
            model_kwargs=claude_model_kwargs
        )

    # --- Build the RAG chain ---
    if risk_rag_chain is None:
        retriever = vectorstore_instance.as_retriever(search_kwargs={"k": 3})
        print("Retriever initialized with top-k search set to 3.")

        def format_docs(docs: list[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful RISK analyst for JPMorgan Chase who analyzes client's profile from financial lens.
                    Based on the following context and the detailed user profile information (if provided),
                    please answer the question accurately and concisely. Analyze the risk aspect of the question.

                    Your response should be formatted as JSON with the following structure:
                    {{
                        "type": "RISK",
                        "results": {{
                            "risk_score": "Provide the risk score on the scale of 1 to 10. It should tell how risky is the query for the client.",
                            "recommended_limit": "Assess the financial situation and suggest a recommended limit in Interger for the asked question. It can be loan or credit card limit",
                            "analysis_title": "Provide a title for the analysis.",
                            "recommendation": "Offer a recommendation based on the analysis.",
                            "key_factors": "List key factors involving risk."
                        }},
                        "evidence": {{
                            "knowledge_graph_reasoning": ["Provide top 3 reasoning based on the knowledge graph"],
                            "supporting_documents": ["List supporting top 3 documents"]
                        }}
                    }}
                    No matter, what your response should only be in the above json format, no additional text. If the answer is not available in the provided information, return an empty json.
                    Pay close attention to any user-specific identifiers (like user ID) and any 'Unstructured Data'
                    notes in the user profile, as these often contain critical insights.

                    Summarize her profile information and any relevant context to provide a comprehensive answer.
                    Do not make up information. Focus on providing relevant details from the context and user profile.
            
                    User Profile from Knowledge Graph:
                    {user_profile_info}
                    """
                ),
                ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
            ]
        )
        print("Prompt template initialized.")
        risk_rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: retriever.invoke(x["question"])
            )
            | RunnablePassthrough.assign(
                context=lambda x: format_docs(x["context"])
            )
            | prompt_template
            | llm_instance
            | StrOutputParser()
        )
        print("RAG chain initialized.")

def clean_query_text(query: str) -> str:
    """Extract the actual query from user input, removing user identifiers."""
    # Remove user name and ID pattern: "Name(ID): query"
    match = re.match(r'^\s*([A-Za-z\s]+)\s*\([PC]\d{3,}\)\s*:\s*(.*)', query, re.IGNORECASE)
    if match:
        return match.group(2).strip()
    
    # Remove standalone ID pattern
    cleaned = re.sub(r'\b[PC]\d{3,}\b', '', query, flags=re.IGNORECASE)
    return cleaned.strip()

def classify_query(query: str) -> str:
    """Classify the query as 'risk' or 'savings'."""
    try:
        # Clean the query to focus on the actual question
        cleaned_query = clean_query_text(query)
        
        print(f"Original query: {query}")
        print(f"Cleaned query: {cleaned_query}")
        
        # Get classification
        result = classification_chain.invoke({"query": cleaned_query})
        
        # Clean and validate result
        classification = result.strip().lower()
        
        # Ensure we return only valid classifications
        if classification in ['risk', 'savings']:
            return classification
        else:
            # Fallback logic based on keywords
            risk_keywords = ['loan', 'credit', 'risk', 'eligibility', 'approval', 'qualify', 'default']
            savings_keywords = ['savings', 'debt-to-income', 'financial health', 'budget', 'spending']
            
            query_lower = cleaned_query.lower()
            
            risk_score = sum(1 for keyword in risk_keywords if keyword in query_lower)
            savings_score = sum(1 for keyword in savings_keywords if keyword in query_lower)
            
            return 'risk' if risk_score > savings_score else 'savings'
            
    except Exception as e:
        print(f"Error in classification: {e}")
        # Default fallback to savings
        return 'savings'
    

def get_fallback_analysis(client_data, analysis_type, query, client_name):
    """Enhanced fallback analysis when RAG Lambda is unavailable"""
    if analysis_type == "loan_analysis":
        # Calculate debt-to-income ratio for better risk assessment
        monthly_income = client_data["income"] / 12
        debt_to_income_ratio = (client_data["debt"] / client_data["income"]) * 100
        
        # Risk scoring based on multiple factors
        credit_risk = max(0, (800 - client_data["credit_score"]) / 20)
        debt_risk = debt_to_income_ratio / 10
        income_stability = 1 if client_data["income"] > 50000 else 2
        
        total_risk = (credit_risk + debt_risk + income_stability) / 3
        max_loan = min(client_data["income"] * 0.4, 50000)
        
        return {
            "metric1_label": "Risk Score",
            "metric1_value": f"{total_risk:.1f}/10",
            "metric2_label": "Max Recommended",
            "metric2_value": f"Rs. {max_loan:,.0f}",
            "analysis_title": "Personal Loan Assessment (Fallback)",
            "analysis_text": f"""**Client Profile:** {client_name} from {client_data['city']}, working in {client_data['occupation']} sector.

**Financial Standing:** With an annual income of Rs. {client_data['income']:,} and current savings of Rs. {client_data['savings']:,}, the client shows {'strong' if total_risk < 4 else 'moderate' if total_risk < 7 else 'weak'} financial stability.

**Risk Assessment:** 
- Credit Score: {client_data['credit_score']} ({'Excellent' if client_data['credit_score'] > 750 else 'Good' if client_data['credit_score'] > 700 else 'Fair'})
- Debt-to-Income Ratio: {debt_to_income_ratio:.1f}%
- Current Debt: Rs. {client_data['debt']:,}

**Recommendation:** {'Approve' if total_risk < 6 else 'Review with caution'} personal loan up to Rs. {max_loan:,.0f}. 

**Savings Goal Impact:** Current goal "{client_data['savings_goal']}" should be considered in loan structuring.

*Note: This is a fallback analysis. RAG system temporarily unavailable.*"""
        }
    
    elif analysis_type == "savings_analysis":
        monthly_savings_potential = (client_data["income"] / 12) - client_data["monthly_expenses"]
        savings_rate = (client_data["savings"] / client_data["income"]) * 100
        
        return {
            "metric1_label": "Savings Rate",
            "metric1_value": f"{savings_rate:.1f}%",
            "metric2_label": "Monthly Potential",
            "metric2_value": f"Rs. {monthly_savings_potential:,.0f}",
            "analysis_title": f"Savings Goal Analysis: {client_data['savings_goal']}",
            "analysis_text": f"""**Current Goal:** {client_data['savings_goal']}
**Progress:** Rs. {client_data['savings']:,} saved with a {savings_rate:.1f}% savings rate.

**Analysis:** {client_name} from {client_data['city']} has {'excellent' if savings_rate > 20 else 'good' if savings_rate > 15 else 'moderate'} savings discipline. 

**Monthly Capacity:** Can potentially save Rs. {monthly_savings_potential:,.0f} per month based on income vs expenses.

**Recommendations:**
- {'Continue current strategy' if savings_rate > 15 else 'Increase savings rate by optimizing expenses'}
- Consider automated savings transfers
- Review goal timeline and adjust monthly targets

*Note: This is a fallback analysis. RAG system temporarily unavailable.*"""
        }
    
    elif analysis_type == "investment_analysis":
        investable_amount = max(0, client_data["savings"] - (client_data["monthly_expenses"] * 6))  # Keep 6-month emergency fund
        risk_appetite = "Conservative" if client_data["age"] > 45 else "Moderate" if client_data["age"] > 30 else "Aggressive"
        
        return {
            "metric1_label": "Investable Amount",
            "metric1_value": f"Rs. {investable_amount:,.0f}",
            "metric2_label": "Risk Profile",
            "metric2_value": risk_appetite,
            "analysis_title": "Investment Portfolio Analysis (Fallback)",
            "analysis_text": f"""**Client Profile:** {client_name}, Age {client_data['age']}, {client_data['occupation']} sector

**Investment Capacity:** Rs. {investable_amount:,.0f} available after maintaining emergency fund.

**Risk Profile:** {risk_appetite} investor based on age and financial stability.

**Sector Insights:** {client_data['occupation']} sector provides {'stable' if client_data['occupation'] in ['Healthcare', 'Education'] else 'dynamic'} income prospects.

**Recommendations:**
- Maintain 6-month emergency fund: Rs. {client_data['monthly_expenses'] * 6:,.0f}
- {'Equity-heavy portfolio (70-80%)' if risk_appetite == 'Aggressive' else 'Balanced portfolio (50-60% equity)' if risk_appetite == 'Moderate' else 'Debt-heavy portfolio (30-40% equity)'}
- Consider tax-saving instruments

*Note: This is a fallback analysis. RAG system temporarily unavailable.*"""
        }
    
    else:
        debt_to_income = (client_data["debt"] / client_data["income"]) * 100
        financial_health = "Excellent" if debt_to_income < 20 else "Good" if debt_to_income < 30 else "Fair"
        
        return {
            "metric1_label": "Debt-to-Income",
            "metric1_value": f"{debt_to_income:.1f}%",
            "metric2_label": "Financial Health",
            "metric2_value": financial_health,
            "analysis_title": "Comprehensive Financial Overview (Fallback)",
            "analysis_text": f"""**Client Overview:** {client_name}, {client_data['age']} years old, based in {client_data['city']}

**Employment:** {client_data['occupation']} sector with annual income of Rs. {client_data['income']:,}

**Financial Health:** {financial_health} with debt-to-income ratio of {debt_to_income:.1f}%

**Current Goals:** Working towards "{client_data['savings_goal']}" with Rs. {client_data['savings']:,} saved

**Key Strengths:**
- Credit Score: {client_data['credit_score']} ({'Excellent' if client_data['credit_score'] > 750 else 'Good' if client_data['credit_score'] > 700 else 'Average'})
- Stable employment in {client_data['occupation']}
- Active savings behavior

**Opportunities:**
- Debt optimization strategies
- Investment portfolio diversification
- Goal-specific financial planning

*Note: This is a fallback analysis. RAG system temporarily unavailable.*"""
        }
    


# Helper functions from savings_rag_lambda_handler.py
def convert_neo4j_int(neo4j_int):
    if isinstance(neo4j_int, (int, float)):
        return neo4j_int
    if hasattr(neo4j_int, 'low') and hasattr(neo4j_int, 'high'):
        return neo4j_int.high * (2**32) + neo4j_int.low
    return neo4j_int

def format_property(key, value, max_key_len, max_value_len):
    formatted_key = key.replace('_', ' ').title()
    if isinstance(value, list):
        formatted_value = ", ".join(map(str, value))
    else:
        formatted_value = str(convert_neo4j_int(value))
    return f"| {formatted_key:<{max_key_len}} | {formatted_value:<{max_value_len}} |"

def query_neo4j_profile(user_id: str = None, user_name: str = None) -> str:
    """
    Queries Neo4j for a user profile based on ID or Name,
    capturing maximum information about the customer, relationships,
    and connected nodes. Returns a comprehensive formatted string in a table format.
    """
    if not neo4j_driver:
        print("Neo4j driver not initialized, cannot query knowledge graph.")
        return "No user profile available from knowledge graph (Neo4j not connected or credentials missing)."

    query_filter = ""
    if user_id:
        query_filter = f"{{id: '{user_id}'}}"
    elif user_name:
        query_filter = f"{{name: '{user_name}'}}"
    else:
        return "No user ID or name provided for Knowledge Graph query."

    query = f"MATCH (c:Customer {query_filter})-[r]-(connectedNode) RETURN c, r, connectedNode"

    profile_sections = []

    try:
        with neo4j_driver.session() as session:
            result = session.run(query)
            records = list(result)

            if not records:
                return f"No profile found in Knowledge Graph for identifier: {user_id if user_id else user_name}."

            # Process customer details once from the first record
            customer = records[0]["c"]
            customer_props = {k: v for k, v in customer.items()}
            customer_props['Labels'] = ", ".join(customer.labels)

            max_key_len = max(len(str(k).replace('_', ' ').title()) for k in customer_props.keys()) if customer_props else 0
            max_value_len = max(len(str(convert_neo4j_int(v))) for v in customer_props.values()) if customer_props else 0

            customer_section_lines = ["--- Customer Profile ---"]
            header = f"| {'Property':<{max_key_len}} | {'Value':<{max_value_len}} |"
            separator = f"| {'-' * max_key_len} | {'-' * max_value_len} |"
            customer_section_lines.extend([header, separator])

            for prop_name, prop_value in customer_props.items():
                customer_section_lines.append(format_property(prop_name, prop_value, max_key_len, max_value_len))

            profile_sections.append("\n".join(customer_section_lines))

            # Now iterate through all records to get connected nodes and relationships
            for record in records:
                relationship = record["r"]
                connected_node = record["connectedNode"]

                connected_node_type_str = ', '.join(connected_node.labels) if connected_node.labels else 'N/A'
                connected_node_name_id = connected_node.get('name', connected_node.get('type', 'N/A'))
                if connected_node.get('id'):
                     connected_node_name_id += f" (ID: {connected_node.get('id')})"
                relationship_type = relationship.type if relationship else "UNKNOWN_RELATIONSHIP"

                connected_entity_props = {
                    "Relationship": relationship_type,
                    "Entity Type": connected_node_type_str,
                    "Entity Name/ID": connected_node_name_id
                }
                for prop_name, prop_value in connected_node.items():
                    if prop_name not in ['name', 'id', 'type']:
                        connected_entity_props[prop_name] = prop_value

                max_key_len_entity = max(len(str(k).replace('_', ' ').title()) for k in connected_entity_props.keys()) if connected_entity_props else 0
                max_value_len_entity = max(len(str(convert_neo4j_int(v))) for v in connected_entity_props.values()) if connected_entity_props else 0

                connected_entity_section_lines = [f"\n--- Connected Entity: {connected_node_type_str} ---"]
                header_entity = f"| {'Property':<{max_key_len_entity}} | {'Value':<{max_value_len_entity}} |"
                separator_entity = f"| {'-' * max_key_len_entity} | {'-' * max_value_len_entity} |"
                connected_entity_section_lines.extend([header_entity, separator_entity])

                for prop_name, prop_value in connected_entity_props.items():
                    connected_entity_section_lines.append(format_property(prop_name, prop_value, max_key_len_entity, max_value_len_entity))

                profile_sections.append("\n".join(connected_entity_section_lines))

            return "\n".join(profile_sections)

    except ServiceUnavailable as e:
        print(f"Neo4j connection lost during query: {e}")
        return "Knowledge Graph temporarily unavailable."
    except Exception as e:
        print(f"Error querying Neo4j: {e}")
        return f"Error fetching profile from Knowledge Graph: {e}"

def extract_user_info_and_clean_query(full_query: str) -> tuple[str, str, str]:
    """
    Extracts user name and ID from the beginning of the query if in 'Name (ID) : Query' format.
    Returns (user_name, user_id, cleaned_query).
    """
    match = re.match(r'^\s*([A-Za-z\s]+)\s*\((P\d{3,})\)\s*:\s*(.*)', full_query, re.IGNORECASE)
    if match:
        user_name = match.group(1).strip()
        user_id = match.group(2).strip().upper()
        cleaned_query = match.group(3).strip()
        print(f"Extracted User Name: {user_name}, ID: {user_id}, Cleaned Query: {cleaned_query}")
        return user_name, user_id, cleaned_query

    id_match = re.search(r'\b([PC]\d{3,})\b', full_query, re.IGNORECASE)
    if id_match:
        user_id = id_match.group(1).upper()
        print(f"Extracted User ID (fallback): {user_id}")
        return None, user_id, full_query

    name_match = re.search(r'(?:user\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', full_query)
    if name_match:
        potential_name = name_match.group(1)
        if potential_name.lower() not in ["context", "question", "answer", "jpmorgan chase", "bedrock", "model"]:
            print(f"Extracted User Name (fallback): {potential_name}")
            return potential_name, None, full_query

    return None, None, full_query

def extract_json_from_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        matches = re.findall(r'\{.*\}', response_text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    return {"type":"UNKNOWN", "results":{}, "evidence":{}}


    # Matches the first { ... } block (non-recursive, so only works with top-level JSONs)
    matches = re.findall(r'\{.*\}', response_text, re.DOTALL)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            return {}

    return {}

    

def call_integrated_rag(client_name, client_id, query, analysis_type):
    """
    Integrated RAG function that replaces the external Lambda call
    """
    try:
        rag_chain=None
        # Initialize RAG components if not already done
        if(analysis_type=="savings"):
            initialize_savings_rag_components()
            rag_chain=savings_rag_chain
        else:
            initialize_risk_rag_components()
            rag_chain=risk_rag_chain

        
        # Format the query like the original Lambda expected
        formatted_query = f"{client_name}({client_id}) : {query}"
        
        # Extract user info and get cleaned query
        user_name, user_id, cleaned_query = extract_user_info_and_clean_query(formatted_query)
        
        user_profile_info = ""
        if user_id:
            print(f"Attempting to fetch user profile for ID: {user_id}")
            user_profile_info = query_neo4j_profile(user_id=user_id)
        elif user_name:
            print(f"Attempting to fetch user profile for Name: {user_name}")
            user_profile_info = query_neo4j_profile(user_name=user_name)
        else:
            user_profile_info = "No specific user identifier found in query to fetch profile."

        if not isinstance(user_profile_info, str):
            user_profile_info = str(user_profile_info)

        print(f"User Profile Info from KG:\n{user_profile_info}")
        print(f"Cleaned Query for RAG: \"{cleaned_query}\"")

        # Prepare the input for the RAG chain
        chain_input = {
            "question": cleaned_query,
            "user_profile_info": user_profile_info
        }

        final_response = rag_chain.invoke(chain_input)
        print(f"\nRAG Response:\n{final_response}")
        parsed_json = extract_json_from_response(final_response)
        # if parsed_json:
        return {
            'success':True,
            'analysis':parsed_json
        }
            
    except Exception as e:
        logger.error(f"Error in integrated RAG: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_fallback_evidence(client_name, client_data):
    kg_insights = [
        f"Employment stability: {client_name} works in {client_data['occupation']} sector showing consistent growth",
        f"Geographic factor: Based in {client_data['city']} with regional economic stability", 
        f"Financial behavior: Current savings goal '{client_data['savings_goal']}' indicates structured financial planning",
        f"Credit profile: Score of {client_data['credit_score']} reflects {'excellent' if client_data['credit_score'] > 750 else 'good'} payment history",
        f"Age demographics: At {client_data['age']} years, falls into {'prime earning' if 25 <= client_data['age'] <= 45 else 'stable career'} age group"
    ]
    
    vector_docs = [
        {"source": "Client Profile - Latest", "text": f"Client based in {client_data['city']}, working in {client_data['occupation']} with goal: {client_data['savings_goal']}"},
        {"source": "Income Verification - Current", "text": f"Verified monthly income of Rs. {client_data['income']/12:,.0f} with stable employment history"},
        {"source": "Credit Report - Recent", "text": f"Credit score of {client_data['credit_score']} with good payment history and low delinquencies"},
        {"source": "Savings Analysis - Latest", "text": f"Current savings: Rs. {client_data['savings']:,} towards goal '{client_data['savings_goal']}'"},
        {"source": f"{client_data['occupation']} Industry Report", "text": f"Sector analysis shows stable growth prospects for {client_data['occupation']} professionals"}
    ]
    
    return kg_insights, vector_docs

try:
    initialize_classifier()
    initialize_risk_rag_components()
    initialize_savings_rag_components()
    logger.info("RAG components initialized successfully")
except Exception as e:
    logger.warning(f"Failed to initialize RAG components: {e}")

def get_client_summary(client_name):
    """Get client summary HTML with enhanced information"""
    if not client_name or client_name not in CLIENT_DATA:
        return ""
    
    client_data = CLIENT_DATA[client_name]
    return f"""
    <div style="background: #f8fbff; border: 1px solid #e6f3ff; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Client ID:</span> <strong>{client_data['id']}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Location:</span> <strong>{client_data['city']}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Annual Income:</span> <strong>Rs. {client_data['income']:,}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Monthly Expenses:</span> <strong>Rs. {client_data['monthly_expenses']:,}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Current Savings:</span> <strong>Rs. {client_data['savings']:,}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Credit Score:</span> <strong>{client_data['credit_score']}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Industry:</span> <strong>{client_data['occupation']}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Savings Goal:</span> <strong>{client_data['savings_goal']}</strong>
        </div>
    </div>
    """

def create_metric_html(label, value):
    """Create styled metric HTML"""
    return f"""
    <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8fbff, #e6f3ff); border-radius: 8px; border-left: 4px solid #4a90e2; margin: 0.5rem 0;">
        <div style="color: #666; font-size: 0.9rem; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: #1f4e79;">{value}</div>
    </div>
    """

def format_kg_insights(kg_insights):
    """Format knowledge graph insights"""
    insights_html = ""
    for insight in kg_insights:
        insights_html += f"""
        <div style="background: #e8f5e8; border-left: 4px solid #4CAF50; padding: 1rem; margin: 0.5rem 0; border-radius: 0 4px 4px 0;">
            <small>{insight}</small>
        </div>
        """
    return insights_html

def format_vector_docs(vector_docs):
    """Format vector database documents"""
    docs_html = ""
    for doc in vector_docs:
        docs_html += f"""
        <div style="background: #fafafa; border-left: 4px solid #4a90e2; padding: 1rem; margin: 0.5rem 0; border-radius: 0 4px 4px 0;">
            <strong style="color: #1f4e79; font-size: 0.85em;">{doc['source']}</strong><br>
            <small>{doc['text']}</small>
        </div>
        """
    return docs_html

def analyze_client(client_name, query):
    """Main analysis function"""
    if not client_name or client_name not in CLIENT_DATA:
        return (
            "Please select a valid client.",
            create_metric_html("Metric 1", "---"),
            create_metric_html("Metric 2", "---"),
            "Please select a valid client to see analysis results.",
            "", ""
        )
    
    client_data = CLIENT_DATA[client_name]
    client_id = client_data.get("id", "UNKNOWN")
    
    # Get client summary
    summary = get_client_summary(client_name)
    
    # Analyze query and get results (now with RAG Lambda integration)
    analysis_type = classify_query(query)




    rag_response = call_integrated_rag(client_name, client_id, query, analysis_type)
    
    if rag_response and rag_response.get('success', False):
        # Use RAG response
        logger.info("Using integrated RAG response for analysis")
        
        # Extract data from RAG response
        analysis_data = rag_response.get('analysis', {})
        results_data = analysis_data.get('results', {})

        if analysis_type == "risk":
            metric1_label = "Risk Score"
            metric1_value = results_data.get('risk_score','N/A')
            metric2_label = "Recommended Limit"
            metric2_value = f"Rs. {results_data.get('recommended_limit', 'N/A')}"
        else:
            metric1_label = "Debt-to-Income Ratio"
            metric1_value = results_data.get('debt_to_income_ratio', 'N/A')
            metric2_label = "Financial Health"
            metric2_value = results_data.get('financial_health', 'N/A')
        
        # Map RAG response to UI format
        results = {
            "metric1_label": metric1_label,
            "metric1_value": metric1_value,
            "metric2_label": metric2_label,  
            "metric2_value": metric2_value,
            "analysis_title": results_data.get('analysis_title', f'{analysis_type} Analysis for {client_name}'),
            "analysis_text": results_data.get('recommendation', 'Analysis from RAG system.')
        }
        evidence_data = analysis_data.get('evidence', {})
        kg_insights = evidence_data.get('knowledge_graph_reasoning', [])
        if isinstance(kg_insights, list):
            kg_insights = kg_insights
        else:
            kg_insights = [str(kg_insights)]
        
        # Format vector docs
        vector_docs_raw = evidence_data.get('supporting_documents', [])
        vector_docs = []
        if isinstance(vector_docs_raw, list):
            for i, doc in enumerate(vector_docs_raw):
                vector_docs.append({
                    "source": f"RAG Document {i+1}",
                    "text": str(doc)
                })
    
    else:
        results= get_fallback_analysis(client_data, analysis_type, query, client_name)
        results = get_fallback_evidence(client_name, client_data)

    
    # Create metric HTML boxes
    metric1_html = create_metric_html(results['metric1_label'], results['metric1_value'])
    metric2_html = create_metric_html(results['metric2_label'], results['metric2_value'])
    
    # Format analysis results
    analysis_content = f"""## {results['analysis_title']}

{results['analysis_text']}

{results['key_factors']}"""
    
    
    # Format evidence sections
    kg_html = format_kg_insights(kg_insights)
    vector_html = format_vector_docs(vector_docs)
    
    return (
        summary,
        metric1_html,
        metric2_html,
        analysis_content,
        kg_html,
        vector_html
    )

custom_css = """
.gradio-container {
    max-width: none !important;
}
.main-header {
    background: linear-gradient(90deg, #1f4e79, #4a90e2);
    padding: 1.5rem 2rem;
    color: white;
    font-size: 1.8rem;
    font-weight: 600;
    margin-bottom: 2rem;
    text-align: center;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.metric-container {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
.section-header {
    color: #1f4e79;
    font-size: 1.4rem;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #e6f3ff;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="RM Intelligence Dashboard", theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.HTML('<div class="main-header">🏦 RM Intelligence Dashboard</div>')
    
    with gr.Row():
        # Column 1: Query Section
        with gr.Column(scale=1):
            gr.HTML('<div class="section-header">Query</div>')
            
            client_dropdown = gr.Dropdown(
                choices=list(CLIENT_DATA.keys()),
                value=list(CLIENT_DATA.keys())[0] if CLIENT_DATA else None,
                label="Select Client",
                container=True
            )
            
            gr.Markdown("**Client Summary**")
            client_summary = gr.HTML()
            
            query_input = gr.Textbox(
                value="What is the risk assessment for a personal loan of Rs50,000?",
                label="Custom Query",
                lines=4,
                placeholder="Enter your specific question about the client",
                container=True
            )
            
            analyze_btn = gr.Button("🔍 Analyze Client", variant="primary", size="lg")
        
        # Column 2: Analysis Results
        with gr.Column(scale=2):
            gr.HTML('<div class="section-header">Analysis Results</div>')
            
            # Metrics row - side by side
            with gr.Row():
                metric1 = gr.HTML(create_metric_html("Risk Score", "2.5"))
                metric2 = gr.HTML(create_metric_html("Max Recommended", "Rs. 30,000"))
            
            # Analysis content
            analysis_results = gr.Markdown("👆 Select a client and click 'Analyze Client' to see results")
        
        # Column 3: Evidence & Reasoning
        with gr.Column(scale=1.2):
            gr.HTML('<div class="section-header">Evidence & Reasoning</div>')
            
            with gr.Accordion("🕸️ Knowledge Graph Reasoning", open=True):
                kg_reasoning = gr.HTML("Knowledge graph insights will be displayed here...")
            
            with gr.Accordion("📄 Vector DB Supporting Documents", open=True):
                vector_documents = gr.HTML("Supporting documents will be shown here...")
    
    # Footer
    gr.Markdown("---")
    gr.Markdown("*Dashboard powered by RAG Pipeline with Knowledge Graph and Vector Search*")
    
    # Event handlers
    def on_analyze_click(client_name, query):
        return analyze_client(client_name, query)
    
    def get_blank_metric():
        return create_metric_html("--", "N/A")
    
    def get_blank_analysis():
        return "👆 Select options and click 'Analyze Client' to see results"
    
    def get_blank_evidence():
        return "Evidence will be displayed after analysis"
    
    
    
    def on_client_change(client_name):
        summary = get_client_summary(client_name)
        return (
            summary, 
            get_blank_metric(),
            get_blank_metric(),
            get_blank_analysis(),
            get_blank_evidence(),
            get_blank_evidence(),
            ""
        )
    
    # Wire up the events
    analyze_btn.click(
        fn=on_analyze_click,
        inputs=[client_dropdown, query_input],
        outputs=[client_summary, metric1, metric2, analysis_results, kg_reasoning, vector_documents]
    )
    
    client_dropdown.change(
        fn=on_client_change,
        inputs=[client_dropdown],
        outputs=[client_summary, metric1, metric2, analysis_results, kg_reasoning, vector_documents, query_input]
    )
    
    # Load initial data when app starts
    demo.load(
        fn=lambda: on_client_change(list(CLIENT_DATA.keys())[0] if CLIENT_DATA else ""),
        outputs=[client_summary, metric1, metric2, analysis_results, kg_reasoning, vector_documents, query_input]
    )

if __name__ == "__main__":
    server_port = int(os.environ.get("AWS_LAMBDA_HTTP_PORT", 8080))
    demo.launch(
        server_name="0.0.0.0", 
        server_port=server_port, 
        share=False,
        show_error=True
    )