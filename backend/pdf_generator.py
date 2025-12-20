"""
PDF Report Generator for Bias Auditor.

Generates comprehensive PDF reports with:
- Executive summary
- Bias origin verdict
- Checkpoint analysis
- Visualizations
- Recommendations
"""
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json


def generate_pdf_report(run_id: str, output_path: Path) -> None:
    """
    Generate a comprehensive PDF report for an audit run.
    
    Args:
        run_id: Run identifier
        output_path: Path to save the PDF
    """
    # Load all artifacts
    from utils import get_artifact_path, get_plots_dir, load_run_config
    
    data_bias_path = get_artifact_path(run_id, "data_bias.json")
    feature_bias_path = get_artifact_path(run_id, "feature_bias.json")
    model_bias_path = get_artifact_path(run_id, "model_bias.json")
    report_path = get_artifact_path(run_id, "bias_origin_report.json")
    
    with open(data_bias_path) as f:
        data_bias = json.load(f)
    with open(feature_bias_path) as f:
        feature_bias = json.load(f)
    with open(model_bias_path) as f:
        model_bias = json.load(f)
    with open(report_path) as f:
        report = json.load(f)
    
    config = load_run_config(run_id)
    plots_dir = get_plots_dir(run_id)
    
    # Create PDF
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title Page
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Bias Checkpoint Auditor", title_style))
    story.append(Paragraph("Comprehensive Audit Report", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph(f"<b>Run ID:</b> {run_id}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", heading_style))
    
    verdict = report['bias_origin_verdict']
    primary_origin = verdict['primary_origin']
    
    # Origin with color coding
    origin_colors = {
        'DATA': '#dc3545',
        'FEATURE': '#ffc107',
        'MODEL': '#28a745',
        'MULTIPLE/UNCLEAR': '#6f42c1'
    }
    origin_color = origin_colors.get(primary_origin, '#6c757d')
    
    story.append(Paragraph(
        f"<b>Primary Bias Origin:</b> <font color='{origin_color}'>{primary_origin}</font>",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(verdict['explanation'], styles['BodyText']))
    story.append(Spacer(1, 0.3*inch))
    
    # Checkpoint Summary Table
    story.append(Paragraph("Checkpoint Summary", heading_style))
    
    checkpoint_data = [['Stage', 'Status', 'Score', 'Issues']]
    for checkpoint in report['checkpoint_summary']:
        issues = ', '.join(checkpoint['flagged_issues'][:3]) if checkpoint['flagged_issues'] else 'None'
        if len(checkpoint['flagged_issues']) > 3:
            issues += '...'
        checkpoint_data.append([
            checkpoint['stage'],
            checkpoint['status'],
            f"{checkpoint['score']:.2f}",
            issues
        ])
    
    checkpoint_table = Table(checkpoint_data, colWidths=[1.5*inch, 1*inch, 1*inch, 3*inch])
    checkpoint_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(checkpoint_table)
    story.append(PageBreak())
    
    # Data Checkpoint Analysis
    story.append(Paragraph("Data Checkpoint Analysis", heading_style))
    story.append(Paragraph(f"<b>Bias Score:</b> {data_bias['overall']['data_bias_score']:.2f}", styles['Normal']))
    story.append(Paragraph(f"<b>Summary:</b> {data_bias['overall']['summary']}", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Add data plots if available
    for attr_name in data_bias['sensitive_attributes'].keys():
        plot_path = plots_dir / f"data_{attr_name}.png"
        if plot_path.exists():
            story.append(Paragraph(f"Distribution: {attr_name}", styles['Heading3']))
            img = Image(str(plot_path), width=5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # Feature Checkpoint Analysis
    story.append(Paragraph("Feature Checkpoint Analysis", heading_style))
    story.append(Paragraph(f"<b>Bias Score:</b> {feature_bias['flags']['feature_bias_score']:.2f}", styles['Normal']))
    story.append(Paragraph(f"<b>Summary:</b> {feature_bias['flags']['summary']}", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Proxy features
    proxy_features = feature_bias.get('proxy_features', {})
    if any(proxy_features.values()):
        story.append(Paragraph("<b>Proxy Features Detected:</b>", styles['Normal']))
        for attr, features in proxy_features.items():
            if features:
                story.append(Paragraph(f"• {attr}: {', '.join(features[:5])}", styles['Normal']))
    
    # Feature heatmap
    heatmap_path = plots_dir / "feature_heatmap.png"
    if heatmap_path.exists():
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("Feature-Attribute Association Heatmap", styles['Heading3']))
        img = Image(str(heatmap_path), width=5*inch, height=4*inch)
        story.append(img)
    
    story.append(PageBreak())
    
    # Model Checkpoint Analysis
    story.append(Paragraph("Model Checkpoint Analysis", heading_style))
    story.append(Paragraph(f"<b>Bias Score:</b> {model_bias['flags']['model_bias_score']:.2f}", styles['Normal']))
    story.append(Paragraph(f"<b>Summary:</b> {model_bias['flags']['summary']}", styles['BodyText']))
    story.append(Spacer(1, 0.2*inch))
    
    # Global metrics
    global_metrics = model_bias['global_metrics']
    story.append(Paragraph(f"<b>Accuracy:</b> {global_metrics['accuracy']:.3f}", styles['Normal']))
    if 'auc' in global_metrics:
        story.append(Paragraph(f"<b>AUC:</b> {global_metrics['auc']:.3f}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Add model plots
    for attr_name in model_bias['by_sensitive_attribute'].keys():
        plot_path = plots_dir / f"model_{attr_name}_metrics.png"
        if plot_path.exists():
            story.append(Paragraph(f"Fairness Metrics: {attr_name}", styles['Heading3']))
            img = Image(str(plot_path), width=5*inch, height=3*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())
    
    # Recommendations
    story.append(Paragraph("Recommended Fixes", heading_style))
    
    fixes = report.get('recommended_fixes', {})
    for stage, recommendations in fixes.items():
        if recommendations:
            story.append(Paragraph(f"<b>{stage}:</b>", styles['Heading3']))
            for i, rec in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {rec}", styles['BodyText']))
            story.append(Spacer(1, 0.2*inch))
    
    # Configuration Details (Appendix)
    story.append(PageBreak())
    story.append(Paragraph("Appendix: Configuration", heading_style))
    story.append(Paragraph(f"<b>Target Column:</b> {config['target_column']}", styles['Normal']))
    
    sens_attrs = ', '.join([sa['name'] for sa in config['sensitive_attributes']])
    story.append(Paragraph(f"<b>Sensitive Attributes:</b> {sens_attrs}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Fairness Thresholds:</b>", styles['Normal']))
    thresholds = config['fairness_thresholds']
    for key, value in thresholds.items():
        story.append(Paragraph(f"• {key}: {value}", styles['Normal']))
    
    # Build PDF
    doc.build(story)
