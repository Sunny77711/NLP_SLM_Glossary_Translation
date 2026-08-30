# 🌾 NLP SLM Glossary Translation

## English → Telugu Agriculture Glossary Translation using IndicTrans2

This project implements an English-to-Telugu agriculture glossary translation pipeline using **IndicTrans2**, a multilingual neural machine translation model developed by **AI4Bharat**.

The system extracts English glossary terms from a PDF document, preprocesses the extracted text, translates it into Telugu using IndicTrans2, and stores the final translations in a CSV file.

---

## 📌 Project Overview

Language barriers can make agricultural information inaccessible to farmers and other stakeholders.

This project addresses this problem by automatically translating English agricultural glossary content into Telugu.

### Pipeline

```text
English Agriculture Glossary (PDF)
            │
            ▼
      Text Extraction
       (pdfplumber)
            │
            ▼
      Preprocessing
            │
            ▼
        IndicTrans2
    English → Telugu
            │
            ▼
   Telugu Glossary (CSV)
```

---

## 🚀 Features

- Extracts text from English glossary PDFs
- Preprocesses extracted glossary entries
- Translates English text into Telugu
- Uses AI4Bharat IndicTrans2
- Supports batch translation
- Automatically detects GPU availability
- Generates a CSV file containing English and Telugu entries
- Designed to run on Google Colab

---

## 🧠 Model Used

**IndicTrans2 En→Indic Distilled 200M**

Model:

```text
ai4bharat/indictrans2-en-indic-dist-200M
```

Source Language:

```text
eng_Latn
```

Target Language:

```text
tel_Telu
```

IndicTrans2 is a multilingual neural machine translation model developed by AI4Bharat for Indian languages.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Translation Model | IndicTrans2 |
| Model Provider | AI4Bharat |
| Deep Learning Framework | PyTorch |
| Transformer Framework | Hugging Face Transformers |
| Translation Toolkit | IndicTransToolkit |
| PDF Processing | pdfplumber |
| Data Processing | Pandas |
| Development Environment | Google Colab |
| Output Format | CSV |

---

## 📂 Repository Structure

```text
NLP_SLM_Glossary_Translation/
│
├── Glossarytranslation.ipynb
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── glossary.pdf
│
└── output/
    └── telugu_glossary.csv
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Sunny77711/NLP_SLM_Glossary_Translation.git
```

Move into the project directory:

```bash
cd NLP_SLM_Glossary_Translation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔐 Hugging Face Authentication

The IndicTrans2 model repository requires Hugging Face authentication and access approval.

### Step 1

Create a Hugging Face account.

### Step 2

Request or accept access to:

```text
ai4bharat/indictrans2-en-indic-dist-200M
```

### Step 3

Create a Hugging Face Read Access Token.

### Step 4

Authenticate:

```python
from huggingface_hub import login

login()
```

Do not upload your Hugging Face token to GitHub.

---

## ▶️ Running the Project

### Google Colab

1. Open `Glossarytranslation.ipynb`
2. Enable GPU:

```text
Runtime → Change runtime type → T4 GPU
```

3. Install dependencies
4. Authenticate with Hugging Face
5. Upload the English glossary PDF
6. Run the notebook cells
7. Download the generated CSV

---

## 📊 Input

The input is an English agriculture glossary PDF.

Example:

```text
Agriculture
Crop
Irrigation
Fertilizer
Harvesting
Soil
```

---

## 📤 Output

The system generates:

```text
telugu_glossary.csv
```

Example:

| English | Telugu |
|---|---|
| Agriculture | వ్యవసాయం |
| Crop | పంట |
| Irrigation | నీటిపారుదల |

---

## ⚡ Hardware

The project can run on CPU.

However, a GPU is recommended for faster translation.

The notebook automatically detects the available device:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 📚 Dependencies

Main dependencies:

```text
PyTorch
Transformers
IndicTransToolkit
Hugging Face Hub
pdfplumber
Pandas
SentencePiece
Sacremoses
Accelerate
```

---

## 🔮 Future Improvements

- Support for additional Indian languages
- Domain-specific agriculture terminology validation
- Translation quality evaluation
- Web-based user interface
- Voice-based glossary queries
- Retrieval-Augmented Generation integration
- Agricultural chatbot integration

---

## 👨‍💻 Author

**V Mokshagnna Bramha Teja**

Roll Number: **CB.SC.U4CSE24353**

---

## 📄 License

This project is created for academic and educational purposes.

---

## 🙏 Acknowledgements

- AI4Bharat
- IndicTrans2
- Hugging Face
- PyTorch