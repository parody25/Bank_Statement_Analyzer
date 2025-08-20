result = {
  "main_title": "Test Bank Report",
  "sections": [
    {
      "title": "1. Customer Summary",
      "content": "This is a brief summary of the customer's financial health. The analysis indicates a stable income but highlights concerns regarding high monthly expenditures."
    },
    {
      "title": "2. Surplus Position",
      "content": "• The average monthly surplus is approximately 150.25.\n• Three out of the last six months showed a negative surplus, which is a potential risk factor."
    },
    {
      "title": "3. Behavioral Insights",
      "content": "• No bounced cheques were observed.\n• Overall Behavior Risk Indicator Score: 2/10, suggesting low-risk financial habits."
    }
  ],
  "footer": {
    "prepared_by": "Automated Analysis System\nTest Bank",
    "date": "2025-08-20"
  }
}

import uuid
from utils.pdf_report_generation import create_templated_pdf

def run_test():
    """
    Runs a standalone test for the PDF generation module.
    """
    print("Starting PDF generation test...")

    # 1. Define the minimal sample data for the report.
    sample_report_data = {
        "main_title": "Test Bank Report",
        "sections": [
            {
                "title": "1. Customer Summary",
                "content": "This is a brief summary of the customer's financial health. The analysis indicates a stable income but highlights concerns regarding high monthly expenditures."
            },
            {
                "title": "2. Surplus Position",
                "content": "• The average monthly surplus is approximately 150.25.\n• Three out of the last six months showed a negative surplus, which is a potential risk factor."
            },
            {
                "title": "3. Behavioral Insights",
                "content": "• No bounced cheques were observed.\n• Overall Behavior Risk Indicator Score: 2/10, suggesting low-risk financial habits."
            }
        ],
        "footer": {
            "prepared_by": "Automated Analysis System\nTest Bank",
            "date": "2025-08-20"
        }
    }

    # 2. Generate a dummy run_id, just like the real endpoint would.
    test_run_id = f"test_{uuid.uuid4()}"

    # 3. Call the PDF generation function directly.
    try:
        create_templated_pdf(report_data=sample_report_data, run_id=test_run_id)
        print("-" * 50)
        print(f"✅ SUCCESS: PDF generation completed.")
        print(f"Check for the file named '{test_run_id}.pdf' in the 'app/reports/' directory.")
        print("-" * 50)
    except FileNotFoundError as fnf_error:
        print("-" * 50)
        print(f"❌ ERROR: A required file was not found.")
        print(f"Details: {fnf_error}")
        print("Please ensure your logo and font files are in the 'app/assets/' directory.")
        print("-" * 50)
    except Exception as e:
        print("-" * 50)
        print(f"❌ ERROR: An unexpected error occurred during PDF generation.")
        print(f"Details: {e}")
        print("-" * 50)

if __name__ == "__main__":
    # This block allows you to run the script directly from the command line.
    run_test()