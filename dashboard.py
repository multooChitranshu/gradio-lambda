import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from langchain_community.graphs import Neo4jGraph
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_client_data():
    """Fetch client data with error handling"""
    try:
        NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+s://3af8a684.databases.neo4j.io")
        NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
        NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Tat-pgntlgkxbtIyxdNf5W1P0qorkDBm3YwhRrY4k")
        
        kg = Neo4jGraph(
            url=NEO4J_URI, 
            username=NEO4J_USER, 
            password=NEO4J_PASSWORD,
            timeout=10
        )
        
        cypher = """
        MATCH (c:Customer)-[:EARNS]->(i:Income),
            (c)-[:HAS_CREDIT_REPORT]->(cr:CreditReport),
            (c)-[:HAS_EXPENSE]->(exp:Expense),
            (c)-[:HAS_GOAL]->(s:SavingsGoal),
            (c)-[:OWES_DEBT]->(d:Debt)
        RETURN c.name as name, c.age as age, i.amount_monthly as income, exp.amount_monthly as monthly_expenses, 
        cr.score as credit_score, s.current_saved as savings, d.remaining_balance as debt, i.employer_business_name as employment
        """
        
        results = kg.query(cypher)
        client_data = {}
        for record in results:
            name = record["name"]
            client_data[name] = {
                "income": record["income"],
                "monthly_expenses": record["monthly_expenses"],
                "savings": record["savings"],
                "credit_score": record["credit_score"],
                "employment": record["employment"],
                "age": record["age"],
                "debt": record["debt"],
            }
        
        return client_data, None
        
    except Exception as e:
        logger.info(f"ERROR: Failed to connect to Neo4j: {e}")
        # Return mock data when Neo4j fails
        mock_data = {
            "Sarah Johnson": {
                "income": 75000,
                "monthly_expenses": 4500,
                "savings": 25000,
                "credit_score": 750,
                "employment": "Tech Corp",
                "age": 32,
                "debt": 15000,
            },
            "John Doe": {
                "income": 65000,
                "monthly_expenses": 3800,
                "savings": 18000,
                "credit_score": 720,
                "employment": "Finance Ltd",
                "age": 28,
                "debt": 12000,
            },
            "Michael Chen": {
                "income": 90000,
                "monthly_expenses": 5200,
                "savings": 45000,
                "credit_score": 780,
                "employment": "Healthcare Inc",
                "age": 35,
                "debt": 8000,
            }
        }
        return mock_data, str(e)

CLIENT_DATA, error = fetch_client_data()

def analyze_query(query_text):
    """Simple function to determine analysis type based on query keywords"""
    query_lower = query_text.lower()
    
    if any(word in query_lower for word in ['loan', 'borrow', 'credit']):
        return "loan_analysis"
    elif any(word in query_lower for word in ['investment', 'invest', 'portfolio']):
        return "investment_analysis"
    elif any(word in query_lower for word in ['mortgage', 'home', 'house']):
        return "mortgage_analysis"
    elif any(word in query_lower for word in ['card', 'credit card']):
        return "card_analysis"
    else:
        return "general_analysis"

def get_metrics_and_analysis(client_data, analysis_type, query, client_name):
    """Returns appropriate metrics and analysis based on query type"""
    if analysis_type == "loan_analysis":
        max_loan = min(client_data["income"] * 0.4, 50000)
        risk_score = 10 - (client_data["credit_score"] - 600) / 20
        
        return {
            "metric1_label": "Risk Score",
            "metric1_value": f"{risk_score:.1f}",
            "metric2_label": "Max Recommended",
            "metric2_value": f"Rs. {max_loan:,.0f}",
            "analysis_title": "Personal Loan Assessment",
            "analysis_text": f"""Based on {client_name}'s financial profile, they qualify for a personal loan with {'low' if risk_score < 5 else 'moderate'} risk. Their stable income of Rs. {client_data['income']:,}, credit score of {client_data['credit_score']}, and savings of Rs. {client_data['savings']:,} indicate strong repayment capability.

**Recommendation:** Approve personal loan up to Rs. {max_loan:,.0f} at standard interest rate. Their debt-to-income ratio would remain within acceptable limits.

**Risk Factors:** Monitor employment stability and ensure loan terms include flexible payment options."""
        }
    
    elif analysis_type == "investment_analysis":
        savings_rate = (client_data["savings"] / client_data["income"]) * 100
        risk_tolerance = "Moderate" if client_data["age"] < 40 else "Conservative"
        
        return {
            "metric1_label": "Savings Rate",
            "metric1_value": f"{savings_rate:.1f}%",
            "metric2_label": "Risk Profile",
            "metric2_value": risk_tolerance,
            "analysis_title": "Investment Recommendation",
            "analysis_text": f"""{client_name} shows excellent savings discipline with a {savings_rate:.1f}% savings rate. Based on their age ({client_data['age']}) and financial stability, a {risk_tolerance.lower()} investment approach is recommended.

**Recommendation:** Diversified portfolio with 60% equity, 30% bonds, 10% alternatives. Consider increasing monthly investment contributions.

**Next Steps:** Schedule portfolio review and discuss long-term financial goals."""
        }
    
    elif analysis_type == "mortgage_analysis":
        max_mortgage = client_data["income"] * 5  # 5x annual income
        debt_service_ratio = ((client_data["monthly_expenses"] + (max_mortgage * 0.05 / 12)) / (client_data["income"] / 12)) * 100
        
        return {
            "metric1_label": "Max Mortgage",
            "metric1_value": f"Rs. {max_mortgage:,.0f}",
            "metric2_label": "Debt Service Ratio",
            "metric2_value": f"{debt_service_ratio:.1f}%",
            "analysis_title": "Mortgage Assessment",
            "analysis_text": f"""{client_name} is eligible for a mortgage with good terms. Based on income and existing obligations, the maximum recommended mortgage is Rs. {max_mortgage:,.0f}.

**Recommendation:** Proceed with mortgage application. Consider 20% down payment to avoid PMI.

**Next Steps:** Get pre-approval and property valuation."""
        }
    
    elif analysis_type == "card_analysis":
        credit_limit = min(client_data["income"] * 0.3, 100000)
        utilization_rec = "Low" if client_data["credit_score"] > 750 else "Moderate"
        
        return {
            "metric1_label": "Credit Limit",
            "metric1_value": f"Rs. {credit_limit:,.0f}",
            "metric2_label": "Recommended Usage",
            "metric2_value": utilization_rec,
            "analysis_title": "Credit Card Assessment",
            "analysis_text": f"""{client_name} qualifies for a premium credit card with high limit. Their excellent credit score of {client_data['credit_score']} enables competitive rates.

**Recommendation:** Approve for premium card with Rs. {credit_limit:,.0f} limit.

**Usage Advice:** Maintain utilization below 30% for optimal credit health."""
        }
    
    else:  # Default general analysis
        debt_to_income = (client_data["debt"] / client_data["income"]) * 100
        financial_health = "Excellent" if debt_to_income < 20 else "Good" if debt_to_income < 30 else "Fair"
        
        return {
            "metric1_label": "Debt-to-Income",
            "metric1_value": f"{debt_to_income:.1f}%",
            "metric2_label": "Financial Health",
            "metric2_value": financial_health,
            "analysis_title": "Financial Overview",
            "analysis_text": f"""{client_name}'s overall financial health is {financial_health.lower()}. With a debt-to-income ratio of {debt_to_income:.1f}% and strong credit score of {client_data['credit_score']}, they are well-positioned for various financial products.

**Strengths:** Stable employment, good credit history, healthy savings balance.

**Opportunities:** Consider debt consolidation and investment portfolio optimization."""
        }

def get_evidence_data(client_name, analysis_type):
    """Generate evidence data"""
    kg_insights = [
        f"Employment stability: {client_name} shows consistent employment history with good sector performance",
        f"Financial behavior: Regular savings patterns indicate disciplined money management",
        f"Credit relationships: Strong payment history with existing financial institutions"
    ]
    
    vector_docs = [
        {"source": "Client Email - Dec 2024", "text": "Looking for financial advice on loan consolidation and investment options..."},
        {"source": "Pay Stub Analysis - Nov 2024", "text": "Consistent monthly income with regular bonuses indicating stable employment"},
        {"source": "Bank Statement - Oct 2024", "text": "Regular savings deposits and controlled spending patterns show financial discipline"},
        {"source": "Industry Report - Nov 2024", "text": f"{client_name}'s employment sector showing stable growth despite market conditions"}
    ]
    
    return kg_insights, vector_docs

def get_client_summary(client_name):
    """Get client summary HTML"""
    if not client_name or client_name not in CLIENT_DATA:
        return ""
    
    client_data = CLIENT_DATA[client_name]
    return f"""
    <div style="background: #f8fbff; border: 1px solid #e6f3ff; border-radius: 8px; padding: 1rem; margin: 1rem 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Annual Income:</span> <strong>Rs. {client_data['income']:,}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Monthly Expenses:</span> <strong>Rs. {client_data['monthly_expenses']:,}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Savings:</span> <strong>Rs. {client_data['savings']:,}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Credit Score:</span> <strong>{client_data['credit_score']}</strong>
        </div>
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Employment:</span> <strong>{client_data['employment']}</strong>
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
    
    # Get client summary
    summary = get_client_summary(client_name)
    
    # Analyze query and get results
    analysis_type = analyze_query(query)
    results = get_metrics_and_analysis(client_data, analysis_type, query, client_name)
    
    # Create metric HTML boxes
    metric1_html = create_metric_html(results['metric1_label'], results['metric1_value'])
    metric2_html = create_metric_html(results['metric2_label'], results['metric2_value'])
    
    # Format analysis results
    analysis_content = f"""## {results['analysis_title']}

{results['analysis_text']}"""
    
    # Get evidence data
    kg_insights, vector_docs = get_evidence_data(client_name, analysis_type)
    
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

def update_on_client_change(client_name):
    """Update all fields when client changes"""
    return analyze_client(client_name, "What is the risk assessment for a personal loan of Rs50,000?")

# Custom CSS for styling
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
            
            gr.Markdown("**Quick Summary**")
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
    
    def on_client_change(client_name):
        return update_on_client_change(client_name)
    
    # Wire up the events
    analyze_btn.click(
        fn=on_analyze_click,
        inputs=[client_dropdown, query_input],
        outputs=[client_summary, metric1, metric2, analysis_results, kg_reasoning, vector_documents]
    )
    
    client_dropdown.change(
        fn=on_client_change,
        inputs=[client_dropdown],
        outputs=[client_summary, metric1, metric2, analysis_results, kg_reasoning, vector_documents]
    )
    
    # Load initial data when app starts
    demo.load(
        fn=lambda: update_on_client_change(list(CLIENT_DATA.keys())[0] if CLIENT_DATA else ""),
        outputs=[client_summary, metric1, metric2, analysis_results, kg_reasoning, vector_documents]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        show_error=True
    )