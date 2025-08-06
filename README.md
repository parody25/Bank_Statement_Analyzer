# Bank_Statement_Analyzer
Building the Project to Analyze the Bank Statement for Retail Bank Customer - HML Loan Process

# Project Structure

my_project/
├── app/
│   ├── main.py
|   ├── controllers/
|   |    └── workflow_definition.py
│   |    └── workflow_runner.py
|   |    └── agents.py
│   ├── models/
│   ├── schemas/
│   └── routers/
│   |    └── document_upload.py
│   |    └── report_analyzer.py
|   └── utils/
|        └── document_preprocessing.py   
├── requirements.txt
└── README.md


## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/adityacodes/flaskcbd.git

```

### 2. Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🧪 Running the App

```bash
fastapi dev app/main.py
```

The API will be available at:  
[http://localhost:8000/docs](http://localhost:8000/docs)

---
## ✅ Todo

- Creating Agents 
- Defining the Workflow Steps Required for Different Analysis 
- Creating Agents 
- Defininig the Inputs and Outputs for each Step 
- Designing the Agent Execution and Process 
---

## 🧑‍💻 Author

Built with ❤️ using FastAPI 
Maintained by CBD Team

---

## 📄 License

This project is licensed under the MIT License.