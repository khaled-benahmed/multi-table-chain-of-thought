Multi-Table Question Answering Project
Overview
This project focuses on extending the WikiTableQuestions dataset and the Chain-Of-Tables (CoT) framework to enable question answering over multiple tables. The goal is to enhance the system's ability to handle complex queries that require integrating data from multiple tables, improving both accuracy and efficiency. The project involves dataset extension, investigation of multi-table processing methods, table selection strategies, and comprehensive evaluation using established metrics.
Project Objectives
The primary objective is to develop a system capable of answering questions by combining data from multiple tables, addressing the limitations of single-table question answering. Key steps include:

Extending the WikiTableQuestions dataset to include multiple tables via vertical partitioning.
Investigating methods to handle multi-table operations, including schema alignment and query planning.
Experimenting with table selection mechanisms to identify relevant tables for specific tasks.
Evaluating the approach using accuracy (per WikiTQ standards), precision, recall, F1-score, and computational efficiency metrics.

Methodology
1. Dataset Extension
The WikiTableQuestions dataset, originally designed for single-table queries, is extended to support multi-table scenarios. This is achieved through vertical partitioning, where a single table is split into multiple tables. For example:

Original Table: Contains columns for Country, Population, and GDP.
Split Tables:
Table 1: Country and Population.
Table 2: Country and GDP.



This approach simulates real-world scenarios where related data is distributed across multiple tables.
2. Multi-Table Processing
The project extends the Chain-Of-Tables (CoT) framework to handle multiple tables by:

Implementing schema alignment techniques (e.g., column normalization, semantic matching) to link related columns across tables.
Developing multi-table query planning strategies, inspired by SQL-based systems, to manage complex queries.
Addressing challenges like table size expansion during joins by exploring optimization techniques such as data pruning and intermediate result compression.

3. Table Selection
To ensure efficiency, the system includes mechanisms to select relevant tables for specific tasks:

Relevance Scoring: Utilizes table content, metadata, schema similarity, and usage patterns to rank tables.
Context-Aware Retrieval: Leverages large language models (LLMs) to interpret query context and dynamically select appropriate tables.
Experiments are conducted with varying table configurations to evaluate selection accuracy and impact on performance.

4. Evaluation
The extended CoT framework is evaluated using:

Accuracy: Measured per WikiTableQuestions (WikiTQ) standards to assess the correctness of answers.
Additional Metrics: Precision, recall, and F1-score to evaluate data retrieval and error minimization.
Computational Efficiency: Analysis of processing time and memory usage to ensure scalability.
Benchmarking: Comparison with state-of-the-art multi-table question answering methods, with statistical significance testing to validate improvements.

Dataset Example
Below is an example of how the dataset is extended:
Original Table



Country
Population
GDP (USD Billion)



USA
331 Million
21,000


Canada
38 Million
1,800


Japan
125 Million
5,000


Extended Multi-Table
Table 1: Population Data



Country
Population



USA
331 Million


Canada
38 Million


Japan
125 Million


Table 2: GDP Data



Country
GDP (USD Billion)



USA
21,000


Canada
1,800


Japan
5,000


Installation
To run the project, follow these steps:

Clone the repository:git clone https://github.com/your-username/multi-table-question-answering.git


Install dependencies:pip install -r requirements.txt


Download the extended WikiTableQuestions dataset (link to be provided).
Run the main script:python main.py



Usage

Dataset Preparation: Use the provided scripts to partition the WikiTableQuestions dataset into multiple tables.
Training/Evaluation: Run experiments using the experiments/ directory scripts to test table selection and query processing.
Evaluation Metrics: Use the provided evaluation scripts to compute accuracy, precision, recall, F1-score, and computational efficiency.

Future Work

Explore horizontal partitioning and related but different tables to further extend the dataset.
Enhance table selection with advanced machine learning models.
Optimize multi-table query processing for larger datasets.
Expand evaluation to include additional benchmarks beyond WikiTQ.

Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bugs.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contact
For questions or feedback, please contact your-email@example.com.
