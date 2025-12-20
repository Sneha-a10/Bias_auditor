"""
Generate PDF report for a specific run.
"""
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from pdf_generator import generate_pdf_report

# Run ID from the demo
run_id = "c0fbb784"

# Output path
output_path = Path(__file__).parent / f"bias_audit_report_{run_id}.pdf"

print(f"Generating PDF report for run: {run_id}")
print(f"Output path: {output_path}")

try:
    generate_pdf_report(run_id, output_path)
    print(f"\n✅ PDF report generated successfully!")
    print(f"📄 Report saved to: {output_path}")
except Exception as e:
    print(f"\n❌ Failed to generate PDF: {e}")
    import traceback
    traceback.print_exc()
