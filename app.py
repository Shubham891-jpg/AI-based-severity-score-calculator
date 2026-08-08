import os
import sys
import gradio as gr
import pandas as pd

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.predict import SeverityPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize predictor
predictor = SeverityPredictor(model_dir="models")
predictor._ensure_embeddings_loaded()

def predict_single_ticket(ticket_text):
    if not ticket_text or not ticket_text.strip():
        return "⚠️ Please enter an IT ticket description.", "None", "0%", "Unknown", ""
    
    result = predictor.predict_single(ticket_text.strip())
    
    if "error" in result:
        return f"❌ Error: {result['error']}", "Error", "0%", "Unknown", ""
    
    score = result.get("severity_score", 50.0)
    category = result.get("severity_category", "Medium")
    confidence = result.get("confidence", 0.0) * 100
    language = result.get("detected_language", "en")
    lang_label = "English" if language == "en" else ("Hindi" if language == "hi" else language)
    processed = result.get("processed_text", "")
    
    score_display = f"{score:.1f} / 100"
    conf_display = f"{confidence:.1f}%"
    
    return score_display, category, conf_display, lang_label, processed

def predict_batch_tickets(batch_text):
    if not batch_text or not batch_text.strip():
        return pd.DataFrame(columns=["#", "Ticket Text", "Severity Score", "Category", "Confidence", "Language"])
    
    lines = [line.strip() for line in batch_text.strip().split("\n") if line.strip()]
    if not lines:
        return pd.DataFrame(columns=["#", "Ticket Text", "Severity Score", "Category", "Confidence", "Language"])
    
    results = predictor.predict_batch(lines)
    
    rows = []
    for i, res in enumerate(results, 1):
        score = res.get("severity_score", 50.0)
        category = res.get("severity_category", "Medium")
        conf = res.get("confidence", 0.0) * 100
        lang = res.get("detected_language", "en")
        lang_label = "English" if lang == "en" else ("Hindi" if lang == "hi" else lang)
        
        rows.append({
            "#": i,
            "Ticket Text": lines[i - 1],
            "Severity Score": f"{score:.1f}",
            "Category": category,
            "Confidence": f"{conf:.1f}%",
            "Language": lang_label
        })
    
    return pd.DataFrame(rows)

# Custom CSS for rich dashboard styling
custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}
.header-box {
    text-align: center;
    padding: 20px 0;
    margin-bottom: 10px;
}
.header-title {
    font-size: 2.2em;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-box {
    border-radius: 10px;
    padding: 15px;
    text-align: center;
}
"""

with gr.Blocks(title="IT Ticket Severity Calculator", css=custom_css, theme=gr.themes.Soft()) as demo:
    with gr.Row(elem_classes=["header-box"]):
        gr.Markdown(
            """
            # 🎫 IT Ticket Severity Calculator
            ### AI-powered Multilingual IT Support Severity Scoring (English & Hindi)
            """
        )
    
    with gr.Tabs():
        with gr.TabItem("🔍 Single Ticket Analyzer"):
            with gr.Row():
                with gr.Column(scale=5):
                    gr.Markdown("#### 📝 Enter IT Ticket Description")
                    input_text = gr.Textbox(
                        label="Ticket Text",
                        placeholder="e.g. All production databases are down, users cannot login...",
                        lines=5
                    )
                    
                    analyze_btn = gr.Button("🚀 Analyze Severity", variant="primary", size="lg")
                    
                    gr.Markdown("#### 💡 Preset Examples")
                    gr.Examples(
                        examples=[
                            ["All production database servers are down, complete system failure across regions."],
                            ["Database is extremely slow, multiple queries timing out on checkout."],
                            ["Office printer is not working, affecting 3 team members."],
                            ["User needs a password reset for their Active Directory account."],
                            ["सर्वर पूरी तरह डाउन है और कोई भी कर्मचारी काम नहीं कर पा रहा है।"]
                        ],
                        inputs=[input_text],
                        label="Click an example to test"
                    )
                
                with gr.Column(scale=4):
                    gr.Markdown("#### 📊 Prediction Output")
                    out_score = gr.Textbox(label="🎯 Severity Score", interactive=False)
                    out_category = gr.Textbox(label="🏷️ Severity Category", interactive=False)
                    out_confidence = gr.Textbox(label="📈 Confidence Score", interactive=False)
                    out_language = gr.Textbox(label="🌍 Detected Language", interactive=False)
                    out_processed = gr.Textbox(label="🧹 Preprocessed Clean Text", interactive=False)
            
            analyze_btn.click(
                fn=predict_single_ticket,
                inputs=[input_text],
                outputs=[out_score, out_category, out_confidence, out_language, out_processed]
            )
        
        with gr.TabItem("📋 Batch Ticket Processor"):
            gr.Markdown("#### 📥 Enter Multiple IT Tickets (One per line)")
            batch_input = gr.Textbox(
                label="Batch Descriptions",
                placeholder="Server 1 down\nPrinter error in floor 2\nPassword reset request\nसर्वर बंद है...",
                lines=7
            )
            batch_btn = gr.Button("⚡ Process Batch Tickets", variant="primary", size="lg")
            
            gr.Markdown("#### 📑 Batch Results Table")
            batch_output = gr.Dataframe(
                headers=["#", "Ticket Text", "Severity Score", "Category", "Confidence", "Language"],
                interactive=False,
                wrap=True
            )
            
            batch_btn.click(
                fn=predict_batch_tickets,
                inputs=[batch_input],
                outputs=[batch_output]
            )
        
        with gr.TabItem("⚙️ Model Specifications & Diagnostics"):
            gr.Markdown(
                """
                ### 🧠 Model Architecture & Details
                - **Algorithm**: Random Forest Regressor (100 Estimators)
                - **Embedding Engine**: `paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional multilingual sentence embeddings)
                - **Quantization**: Dynamic Int8 PyTorch Quantization (optimizes RAM below 200 MB)
                - **Languages Supported**: English (`en`) & Hindi (`hi`)
                - **Score Calibration**: 10 (Minimal) to 100 (Critical Outage)
                - **Training Dataset**: 2,936 Real-world IT support tickets
                """
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
