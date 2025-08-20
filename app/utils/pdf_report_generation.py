import os
from fpdf import FPDF
from pathlib import Path
from datetime import datetime

# Define base paths
APP_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = APP_DIR / "reports"
ASSETS_DIR = APP_DIR / "assets"

# Ensure directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR / "images", exist_ok=True)
os.makedirs(ASSETS_DIR / "fonts", exist_ok=True)


class PDF(FPDF):
    def header(self):
        # Logo: path, x, y, width
        logo_path = ASSETS_DIR / "images" / "logo.jpeg"
        if logo_path.exists():
            self.image(str(logo_path), 10, 8, 33)
        
        self.set_font('DejaVu', 'B', 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, self.title, 0, 0, 'C')
        # Line break
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_templated_pdf(report_data: dict, run_id: str):
    """
    Generates a templated PDF report from a structured dictionary and saves it.
    """
    pdf = PDF()
    
    # --- FONT SETUP ---
    font_path = ASSETS_DIR / "fonts" / "DejaVuSans.ttf"
    font_path_bold = ASSETS_DIR / "fonts" / "DejaVuSans-Bold.ttf"
    font_path_italic = ASSETS_DIR / "fonts" / "DejaVuSans-Oblique.ttf"
    if not font_path.exists():
        raise FileNotFoundError(f"Font file not found at {font_path}")
    pdf.add_font('DejaVu', '', str(font_path), uni=True)
    pdf.add_font('DejaVu', 'B', str(font_path).replace(".ttf", "-Bold.ttf"), uni=True) # Assumes Bold variant exists
    pdf.add_font('DejaVu', 'I', str(font_path_italic), uni=True) # Register the Italic font
    
    # Set document metadata
    pdf.set_title(report_data.get("main_title", "Financial Report"))
    pdf.set_author("Standard Bank")

    pdf.add_page()
    
    # --- RENDER SECTIONS ---
    for section in report_data.get("sections", []):
        # Section Title
        pdf.set_font('DejaVu', 'B', 14)
        pdf.set_text_color(0, 84, 166) # A nice blue color
        pdf.cell(0, 10, section["title"], 0, 1, 'L')
        
        # Underline for the title
        pdf.set_draw_color(0, 84, 166)
        pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + 190, pdf.get_y())
        pdf.ln(5) # Spacing after the line
        
        # Section Content
        pdf.set_font('DejaVu', '', 11)
        pdf.set_text_color(0, 0, 0) # Black
        
        # Clean up content for better rendering (e.g., bullet points)
        content = section["content"].replace('•', '  - ')
        
        pdf.multi_cell(0, 7, content)
        pdf.ln(10) # Spacing between sections
    
    # --- FOOTER CONTENT ---
    footer_data = report_data.get("footer", {})
    if footer_data:
        pdf.set_font('DejaVu', '', 10)
        pdf.ln(15)
        pdf.multi_cell(0, 5, f"Prepared by:\n{footer_data.get('prepared_by', '')}")
        
        # Use current date if not provided
        report_date = footer_data.get('date', datetime.now().strftime("%Y-%m-%d"))
        if "[Insert Date]" in report_date:
            report_date = datetime.now().strftime("%Y-%m-%d")
        
        pdf.ln(5)
        pdf.cell(0, 5, f"Date: {report_date}")

    # --- SAVE THE FILE ---
    output_filename = f"{run_id}.pdf"
    output_path = REPORTS_DIR / output_filename
    pdf.output(name=str(output_path))
    
    print(f"Successfully generated templated report to: {output_path}")