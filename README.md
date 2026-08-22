# 🧬 EvoMind

**An agentic AI system that evolves its own problem-solving strategy.**

Not another chatbot. Not another RAG system. EvoMind is a self-improving AI that modifies and optimizes its own analytical workflow based on performance — using structured mutation operators, multi-objective fitness evaluation, and persistent cross-task learning.

Give it a task like *"analyze this unfamiliar dataset and discover useful patterns"* and it runs an evolutionary loop:

```text
plan a strategy → execute it → evaluate results → reflect → mutate → repeat
```

Every strategy it tries — its lineage, mutations, and scores — is written to a persistent SQLite memory. The next time it sees a similar task, it starts from what already worked.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧬 **Structured Evolution** | 6 mutation operators (add, remove, swap, tune, reorder, crossover). |
| 📊 **16 Analytical Steps** | Random Forests, Isolation Forests, PCA, DBSCAN, Mutual Info, and more. |
| 🧠 **Persistent Memory** | SQLite-backed strategy recall with cross-task transfer learning (embeddings). |
| 📈 **Multi-Objective Fitness** | Scores on insight depth, coverage, efficiency, and novelty. |
| 🔌 **Plugin System** | Drop a `.py` file in `plugins/` to add custom steps seamlessly. |
| 🌐 **Real-Time Dashboard** | FastAPI + React (Vite) dashboard showing live evolution and score trajectories. |
| 📁 **Multi-Format Data** | CSV, Parquet, JSON, JSONL, Excel, TSV, SQL databases, and HTTP URLs. |
| 🤖 **Multi-Provider LLM** | Groq (free default), HuggingFace, and Anthropic — switch via `.env`. |

---

## 🚀 How to Run the Project

It's super easy to get started.

### 1. Prerequisites
- Python 3.10+
- Node.js (for the dashboard)

### 2. Installation
Clone the repository, create a virtual environment, and install the dependencies:
```bash
# Create and activate a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1
# Mac/Linux
source venv/bin/activate

# Install all requirements
pip install -r requirements.txt
```

### 3. API Keys
Create a `.env` file in the root directory (next to this README) and add your API keys. By default, the project uses **Groq** for fast, free inference:
```env
EVOMIND_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
```
*(You can also use `EVOMIND_PROVIDER=anthropic` or `huggingface` if you provide the respective keys).*

---

### Option A: Run the Real-Time Dashboard (Recommended!)
The dashboard is the best way to experience EvoMind. It requires two terminal windows.

**Terminal 1: Start the Backend API**
```bash
# Ensure your virtual environment is active
python -m evomind.api
```

**Terminal 2: Start the Frontend React App**
```bash
# Navigate to the dashboard directory
cd evomind/dashboard

# Install NPM dependencies (only needed the first time)
npm install

# Start the dev server
npm run dev
```
Open **http://localhost:3000** in your browser. You can upload a dataset (e.g., `data/sample_sales.csv`) and watch the AI evolve!

---

### Option B: Run the Command-Line Interface (CLI)
If you just want to run an analysis straight from the terminal and get a text output/report:

```bash
# Ensure your virtual environment is active
python -m evomind.main --data data/sample_sales.csv --task "Analyze sales trends and find correlations" --iterations 5 --threshold 0.85 --save --report
```

---

## 🏗️ Architecture

```text
planner → executor → evaluator → reflector ─┬─→ planner   (score too low, iterations left)
                                             └─→ END       (threshold hit, or max iterations)
```
1. **Planner**: Uses the LLM to propose a strategy (iter 0) or selects mutation operators (iter 1+) based on past failures.
2. **Executor**: Safely runs the proposed analytical steps against the dataset using pandas/scikit-learn/scipy.
3. **Evaluator**: Calculates multi-objective scoring (insight, coverage, efficiency, novelty) + LLM judgment.
4. **Reflector**: Saves results to memory, tracks lineage, and decides whether to continue or stop.

## 🧪 Running Tests
EvoMind comes with a suite of 51 fully offline, deterministic tests.
```bash
python -m pytest tests/ -v
```

## License
MIT
