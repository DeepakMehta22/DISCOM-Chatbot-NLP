# Electricity Conversational Chatbot using Traditional NLP

A rule-based + ML-powered conversational chatbot built for an electricity utility provider. The bot is designed to handle common customer service interactions — greetings, account information lookups, new connection registration, payment & refund queries, and complaint registration — using **traditional NLP techniques** (TF-IDF + Neural Network for intent classification, regex-based field validation, spaCy-based entity extraction, and TextBlob-based sentiment analysis).

---

## 📌 Objective

To build an electricity conversational chatbot with the following capabilities:

1. **Greetings** – respond to common greeting messages.
2. **Account at a Glance (Existing Customer)** – current month's bill, latest meter reading, consumption pattern over the last 12 months, and month-on-month (MoM) consumption growth.
3. **Account Registration (New Customer)** – collect Name, Mobile, PIN Code, Aadhaar, and PAN, run sanity checks on Aadhaar and PAN, and confirm that the application has been received.
4. **Payment & Refund** – handle queries such as:
   - Why a payment isn't reflecting
   - Money deducted but no confirmation received
   - Money deducted twice
   - Refunds / adjustments in the next billing cycle
5. **Complaints** – handle:
   - Power cut (collect CA Number, PIN Code, duration, and share expected restoration time)
   - Faulty meter
   - High bill due to incorrect reading or calculation error

---

## 🧠 Approach

### Step 1 — Intent Classification
- 12 intents were defined, each with 10+ training examples (`Intents-examples.xlsx`).
- **Pre-processing**: stopword removal, removing numerals, lowercasing, and lemmatization.
- **Vectorization**: training examples are converted into numerical vectors using a **TF-IDF Vectorizer**.
- **Model**: a **Neural Network (Keras/TensorFlow Sequential model)** is trained on the TF-IDF vectors to classify the intent of an incoming user message.

The 12 intents are:

| Intent | Description |
|---|---|
| `greet` | Greeting messages |
| `monthlyBill` | Current month's bill amount |
| `meterReading` | Latest meter reading |
| `consumptionPattern` | Consumption trend over last 12 months |
| `momGrowth` | Month-on-month consumption growth |
| `paymentReceipt` | Payment receipt status |
| `paymentReflection` | Payment not reflecting in account |
| `twiceDeduction` | Amount deducted twice |
| `newConnection` | New connection / account registration |
| `powerCut` | Power outage complaint |
| `faultyMeter` | Faulty meter complaint |
| `highBill` | High / incorrect bill complaint |

### Step 2 — Input Validation Functions
Regex / rule-based validation functions were built for:

| Field | Rule |
|---|---|
| **Customer ID** | 6-digit number, validated against the dummy customer database (`user_data.xlsx`) |
| **PAN Card No.** | 10-character alphanumeric code — first 5 and last character are letters, the middle 4 are digits |
| **Mobile No.** | 10-digit number starting with 7, 8, or 9 |
| **PIN Code** | 6-digit code |
| **Aadhaar Card No.** | 12-digit number |

### Step 3 — Query Response Functions
For each of the 12 intents, a dedicated response function was created. Each function:
- First checks an in-memory `memory` dictionary for the user's account details (id, name, mobile, etc.).
- If no data is available, prompts the user for the required information (e.g. Customer ID, name) to either fetch existing account details or begin a new account registration.

### Additional Features
- **Sentiment check** — for messages with a "neutral" predicted intent, the polarity of the message is analyzed; if the polarity score is strongly positive/negative (`> 0.2` or `< -0.2`), a sentiment-appropriate response is generated.
- **`clear` keyword** — clears all data stored in the `memory` dictionary at any point during the conversation.
- **`quit` keyword** — allows the user to exit the conversation at any time.
- **Entity extraction** — uses a custom-trained spaCy NER model to extract names, IDs, mobile numbers, and PIN codes directly from free-form user input.
- **Spelling correction** — uses a vocabulary-based correction algorithm (`pyspellchecker`) to fix misspelled words in user queries before intent classification.

---

## 🛠️ Tech Stack

- **Python**
- **pandas / numpy** – data handling
- **NLTK** – stopword removal, tokenization, lemmatization, stemming
- **scikit-learn** – TF-IDF vectorization, train/test split, label binarization, evaluation metrics
- **TensorFlow / Keras** – Neural Network for intent classification
- **spaCy** (`en_core_web_sm`) + **spaCy custom NER training** – named entity recognition
- **TextBlob / spacytextblob** – sentiment polarity analysis
- **pyspellchecker** – spelling correction

---

## 📁 Repository Structure

```
Electricity-Chatbot-NLP/
├── Chatbot_v2.ipynb        # Main notebook: data prep, model training, chatbot logic
├── Intents-examples.xlsx   # Training data – intents and example utterances
├── user_data.xlsx          # Dummy customer database used for validation & responses
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Electricity-Chatbot-NLP
```

### 2. Install dependencies
```bash
pip install pandas numpy nltk scikit-learn tensorflow textblob pyspellchecker spacy spacytextblob
python -m spacy download en_core_web_sm
```

The notebook also downloads required NLTK corpora at runtime (`stopwords`, `punkt`).

### 3. Run the notebook
Open `Chatbot_v2.ipynb` in Jupyter / Google Colab and run the cells sequentially:

1. **Import libraries** and download NLTK/spaCy resources.
2. **Load training data** (`Intents-examples.xlsx`) and **customer database** (`user_data.xlsx`) — keep these files in the same directory as the notebook.
3. **Preprocess** the training examples and build the vocabulary.
4. **Vectorize** examples using TF-IDF and encode intent labels.
5. **Train** the Neural Network intent classifier.
6. **Train** the custom spaCy NER model for entity extraction.
7. **Run the chatbot** in the final cell — it will prompt for user input (`YOU:`) in a loop.

### 4. Chat with the bot
Once the final cell is running, type messages like:

```
YOU: hi
YOU: what's my current bill?
YOU: I want a new connection
YOU: clear
YOU: quit
```

---

## 📊 Data Files

- **`Intents-examples.xlsx`** — Two columns: `Intents` and `Examples`. Contains sample user utterances mapped to each of the 12 intents, used to train the TF-IDF + Neural Network classifier.
- **`user_data.xlsx`** — A dummy customer database with fields such as `id`, `month_bill`, `meter_reading`, `consumption_pattern`, `mom`, `name`, `mobile`, `pin_code`, `aadhaar`, `pan`, and complaint/ticket details, used to validate Customer IDs and generate account-specific responses.

> ⚠️ Note: All customer data in `user_data.xlsx` is **dummy/synthetic data** created solely for testing purposes.

---

## 🔮 Future Improvements
- Replace TF-IDF + dense NN with transformer-based intent classification (e.g. fine-tuned BERT/DistilBERT).
- Persist conversation memory and ticket data to a real database instead of an in-memory dictionary / Excel file.
- Add a web/chat UI (e.g. Streamlit, Flask, or a messaging platform integration) instead of console-based interaction.
- Expand intent coverage and add multi-turn dialogue management for more complex queries.
