# Multi-Table Question Answering

This project extends the WikiTableQuestions dataset and the Chain-Of-Tables (CoT) framework to enable question answering over **multiple tables**. It enhances the system's ability to handle **complex queries** that require integrating data from several sources, improving both accuracy and efficiency.

---

## 🚀 Project Objectives

The primary objective is to develop a system capable of answering questions by combining data from multiple tables, addressing the limitations of single-table question answering.

### Key Steps:

- 📚 Extend the WikiTableQuestions dataset using vertical partitioning.
- 🔧 Develop methods for multi-table processing (schema alignment, query planning).
- 🧠 Implement table selection mechanisms using content and context-aware strategies.
- 📊 Evaluate performance using metrics like accuracy, precision, recall, F1-score, and efficiency.

---

## 🛠 Methodology

### 1. Dataset Extension

The original WikiTableQuestions dataset is adapted to simulate multi-table scenarios using **vertical partitioning**:

**Example:**

| Country | Population | GDP (USD Billion) |
|---------|------------|-------------------|
| USA     | 331 Million| 21,000            |
| Canada  | 38 Million | 1,800             |
| Japan   | 125 Million| 5,000             |

Becomes:

**Table 1: Population Data**

| Country | Population |
|---------|------------|
| USA     | 331 Million|
| Canada  | 38 Million |
| Japan   | 125 Million|

**Table 2: GDP Data**

| Country | GDP (USD Billion) |
|---------|-------------------|
| USA     | 21,000            |
| Canada  | 1,800             |
| Japan   | 5,000             |

---

### 2. Multi-Table Processing

- 🔗 **Schema Alignment:** Column normalization and semantic matching to align related columns.
- 📄 **Query Planning:** SQL-inspired multi-table strategies for complex question answering.
- ⚙️ **Optimization:** Handle table size expansion using pruning and intermediate result compression.

---

### 3. Table Selection

To ensure scalability and accuracy:

- ⭐ **Relevance Scoring:** Based on table content, metadata, schema similarity, and usage history.
- 🤖 **Context-Aware Retrieval:** Uses LLMs to interpret query context and dynamically select relevant tables.

---

### 4. Evaluation

- ✅ **Accuracy:** Measured according to WikiTableQuestions standards.
- 📈 **Additional Metrics:** Precision, recall, F1-score.
- 🧮 **Computational Efficiency:** Runtime and memory usage.
- 🔬 **Benchmarking:** Compared against state-of-the-art multi-table QA systems with significance testing.

---
