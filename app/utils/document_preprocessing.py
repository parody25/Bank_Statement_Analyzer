import os
from dotenv import load_dotenv
from pathlib import Path
#from pydantic import BaseModel
from pydantic import BaseModel, Field
from pydantic import ValidationError
import json
from typing import Dict
#LlamaParse
from llama_parse import LlamaParse
#LlamaIndex Imports
from llama_index.core import SimpleDirectoryReader
#LlamaIndex OpenAI Settings
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from langchain.schema.runnable import Runnable

from llama_index.core import Settings


#Langchain Imports 
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


#Pandas Imports to handle DataFrames and Tables
import pandas as pd
import numpy as np




_ = load_dotenv()  # Load environment variables from .env file

BASE_DIR: Path = Path(__file__).resolve().parent.parent
UPLOAD_DIR = f"{BASE_DIR}/uploads/tables_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Looping Multiple Files and Parsing the Documents for particular Company 
parser = LlamaParse(api_key=os.getenv("LLAMA_API_KEY"),
                #structured_output_json_schema_name="bank_statement",structured_output=True
                #result_type="markdown",
                parsing_instruction = """While Parsing the Tables from the documents ensure that the column name is maintained across all the pages even if column name is mentioned only in the 1st page of the document.""",
                result_type="json",
                merge_tables_across_pages_in_markdown=False,
                compact_markdown_table=False,
                continuous_mode=False,
                verbose=True,
                #page_prefix="START OF PAGE: {pageNumber}\n",page_suffix="\nEND OF PAGE: {pageNumber}",
                )

class AlignedResponse(BaseModel):
    aligned_table: bool
    rename: Dict[str, str]

SYSTEM_PROMPT = """
You are a data quality assistant specialized in verifying tabular data structure and standardizing column names for financial transaction data.

Your output must be a JSON with two keys:
- standardized_table: boolean (true if the table is standardized, false if any corrections are needed)
- rename: a dictionary mapping old column names to their standardized names
"""

# Supported file types mapped to the parser
FILE_EXTRACTOR = {
    ".pdf": parser,
    ".png": parser,
    ".jpg": parser,
    ".jpeg": parser,
    ".xlsx": parser,
    ".xls": parser,
    ".csv": parser
}

async def preprocess_documents(file_path: str):
    """
    Parse all supported files in the given directory using LlamaParse.

    Args:
        file_path (str): Absolute or relative path to the uploads directory.

    Returns:
        List[Document]: List of parsed document objects.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Directory not found: {file_path}")
    
    
    file_path = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg") or ".pdf" in file or ".xlsx" in file or ".xls" in file or ".csv" in file]
    print(file_path)
    
    # Specific to get the JSON Output from the LlamaParse at Page Level 
    documents = await parser.aget_json(file_path)

    #documents = await SimpleDirectoryReader(
    #    input_dir=file_path,
    #    file_extractor=FILE_EXTRACTOR
    #).aload_data()
    #for doc in documents:
    try:
        parser.get_tables(documents, UPLOAD_DIR)
    except Exception as e:
        print(f"Error extracting tables: {e}")

    # Debugging the Preprocessed documents
    print(f"Documents Preprocessed from {file_path}")
    #print(documents)
    print(len(documents))
    for doc in documents:
        print(type(doc))
    return documents

def looks_like_header(row):
    """
    Returns True if the row looks like a header (e.g., all string, no numbers).
    """
    return all(isinstance(val, str) or str(val).isalpha() for val in row)  # Heuristic

#def looks_like_header(row):
#    num_text_like = sum(str(val).isalpha() for val in row)
#    return num_text_like >= len(row) * 0.8  # 80%+ of values are alphabetic

def post_process_tables():
    """
    Concatenates all CSV files in the given directory into a single DataFrame.

    Returns:
        pd.DataFrame: Concatenated DataFrame from all CSV files.
    """

    csv_files = [file for file in os.listdir(UPLOAD_DIR) if file.endswith('.csv')]
    df_list = []
    header_columns = None  # Store header from first file
    
    for idx, file in enumerate(csv_files):
        file_path = os.path.join(UPLOAD_DIR, file)

        if idx == 0:
            # Read the first file normally (with header)
            df = pd.read_csv(file_path)
            header_columns = df.columns
        else:
            # Read subsequent files with no header and assign same column names
            df = pd.read_csv(file_path, header=None)
            df.columns = header_columns 

            # Check if first row matches header (i.e., it's actually a header row)
            #if df.iloc[0].equals(pd.Series(header_columns)):
            #    df = df[1:]  # Remove the header row
            first_row = df.iloc[0]
            if looks_like_header(first_row):
                df = df[1:]  # Skip it
        
        df_list.append(df)
    if df_list:
        combined_df = pd.concat(df_list, ignore_index=True)
        # Standardize the column names using the GenAI take the column name and replace with the standard column names
        #combined_df.columns = [col.strip().lower().replace(' ', '_') for col in combined_df.columns]
        markdown_df = combined_df.to_markdown(index=False)
        with open('app/prompts/Column_name_change.txt', 'r', encoding='utf-8') as f:
            user_prompt_template = f.read()
        prompt = PromptTemplate.from_template(user_prompt_template)
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        #structured_llm = llm.with_structured_output(StandardizeResponse)
        structured_llm = llm.with_structured_output(method="json_mode",schema=AlignedResponse.model_json_schema())
        chain: Runnable = prompt | structured_llm
        response = chain.invoke({
    "table_str": markdown_df
})
        print("Model response:", response)

        try: 
            #parsed_response = json.loads(str(response))
            #response = StandardizeResponse(parsed_response)
            if response["aligned_table"]: 
                combined_df.rename(columns=response["rename"], inplace=True)

                #Changing the DataType for the Columns 
                print(combined_df.dtypes)
                cols_type = ['TransactionDate','Credit', 'Debit', 'Balance']
                cols_type = ["TransactionDate"]
                if "TransactionDate" in combined_df.columns:
                    combined_df['TransactionDate'] = pd.to_datetime(combined_df['TransactionDate'], errors='coerce', dayfirst=True)
                    print("TransactionDate converted to datetime:", combined_df['TransactionDate'].dtypes)
                    print("TransactionDate values:", combined_df['TransactionDate'].head())
                if "Credit" in combined_df.columns:
                    combined_df['Credit'] = pd.to_numeric(combined_df['Credit'], errors='coerce').astype(float).fillna(0.0)
                    print("Credit converted to float:", combined_df['Credit'].dtypes)
                    print("Credit values:", combined_df['Credit'].head())
                if "Debit" in combined_df.columns:
                    combined_df['Debit'] = pd.to_numeric(combined_df['Debit'], errors='coerce').astype(float).fillna(0.0)
                    print("Debit converted to float:", combined_df['Debit'].dtypes)
                    print("Debit values:", combined_df['Debit'].head()) 
                
                if "Balance" in combined_df.columns:
                    combined_df['Balance'] = pd.to_numeric(combined_df['Balance'], errors='coerce').astype(float).fillna(0.0)
                    print("Balance converted to float:", combined_df['Balance'].dtypes)
                    print("Balance values:", combined_df['Balance'].head()) 
                # Add a new column 'Type' based on Credit and Debit values
                if "Credit" in combined_df.columns and "Debit" in combined_df.columns:
                    combined_df['Type'] = None  # ensures object dtype for strings
                    combined_df.loc[combined_df['Credit'].fillna(0) > 0, 'Type'] = 'credit'
                    combined_df.loc[combined_df['Debit'].fillna(0) > 0, 'Type'] = 'debit'
                #if "Credit" in combined_df.columns and "Debit" in combined_df.columns:
                #    combined_df['Type'] = np.where(
                #        combined_df['Credit'].fillna(0) > 0, 'credit',
                #        np.where(combined_df['Debit'].fillna(0) > 0, 'debit', None)
                #    )
                print("Data types after conversion:", combined_df.dtypes)
            else:
                print("Table is not standardized. Please check the rename mapping.")
        except ValidationError as e:
            print(f"Validation error: {e}")
            print("Response was not valid JSON or did not match the expected schema.")
            return combined_df
        

    return combined_df






