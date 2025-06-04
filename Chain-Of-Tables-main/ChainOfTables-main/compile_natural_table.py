import json
import copy
from datasets import load_dataset
import os
import random

def split_table_vertically(table, question, answers, id_str):
    """
    Split a table vertically into two related tables, preserving one common column.
    
    Args:
        table: Original table dictionary with 'header' and 'rows'
        question: The question associated with the table
        answers: The answers associated with the question
        id_str: Unique identifier for this instance
        
    Returns:
        A dictionary in the multi-table format expected by compile_natural_table
    """
    header = table['header']
    rows = table['rows']
    
    # Ensure we have enough columns to split meaningfully
    if len(header) < 3:
        # For very small tables, duplicate them instead of splitting
        return create_duplicated_tables(table, question, answers, id_str)
    
    # Choose the first column as the common join key in most cases
    # This is typically an identifier (name, ID, etc.)
    join_column_idx = 0
    
    # Determine split point (approximately half the columns, excluding the join column)
    remaining_cols = len(header) - 1
    split_point = 1 + (remaining_cols // 2)  # +1 because we're starting after the join column
    
    # Create headers for the two tables (both including the join column)
    header_1 = [header[join_column_idx]] + header[1:split_point]
    header_2 = [header[join_column_idx]] + header[split_point:]
    
    # Create rows for the two tables
    rows_1 = []
    rows_2 = []
    
    for row in rows:
        # Extract the join column value
        join_value = row[join_column_idx]
        
        # Create the rows for each table
        row_1 = [join_value] + row[1:split_point]
        row_2 = [join_value] + row[split_point:]
        
        rows_1.append(row_1)
        rows_2.append(row_2)
    
    # Create meaningful table names based on the headers
    # Use the most descriptive column names from each table
    table1_name = get_table_name(header_1, 1)
    table2_name = get_table_name(header_2, 2)
    
    # Create the multi-table format expected by compile_natural_table
    multi_table = {
        "id": id_str,
        "question": question,
        "answers": answers,
        "tables": [
            {
                "name": table1_name,
                "header": header_1,
                "rows": rows_1
            },
            {
                "name": table2_name,
                "header": header_2,
                "rows": rows_2
            }
        ]
    }
    
    return multi_table

def create_duplicated_tables(table, question, answers, id_str):
    """
    For tables that are too small to split meaningfully, create two related tables
    by duplicating with slight variations.
    """
    header = table['header']
    rows = table['rows']
    
    # Ensure we have at least 2 columns
    if len(header) < 2:
        # Add a dummy column if there's only one
        header.append("Additional Info")
        for i in range(len(rows)):
            rows[i].append("N/A")
    
    # Use first column as join key
    join_column_idx = 0
    
    # Create two tables with different additional columns
    header_1 = [header[join_column_idx], header[1]]
    if len(header) > 2:
        header_1.append(header[2])
    
    header_2 = [header[join_column_idx]]
    for i in range(1, len(header)):
        if i not in [1, 2]:  # Exclude columns already in table 1
            header_2.append(header[i])
    
    # If table 2 only has the join column, add a dummy column
    if len(header_2) == 1:
        header_2.append("Additional Info")
    
    # Create rows
    rows_1 = []
    rows_2 = []
    
    for row in rows:
        # Extract the join column value
        join_value = row[join_column_idx]
        
        # Create row 1
        row_1 = [join_value, row[1]]
        if len(header_1) > 2:
            row_1.append(row[2] if len(row) > 2 else "N/A")
        
        # Create row 2
        row_2 = [join_value]
        for i in range(1, len(header)):
            if i not in [1, 2]:  # Exclude columns already in table 1
                row_2.append(row[i] if i < len(row) else "N/A")
        
        # If table 2 only has the join column, add a dummy value
        if len(row_2) == 1:
            row_2.append("N/A")
        
        rows_1.append(row_1)
        rows_2.append(row_2)
    
    # Create table names
    table1_name = get_table_name(header_1, 1)
    table2_name = get_table_name(header_2, 2)
    
    # Create the multi-table format
    multi_table = {
        "id": id_str,
        "question": question,
        "answers": answers,
        "tables": [
            {
                "name": table1_name + "Split1",
                "header": header_1,
                "rows": rows_1
            },
            {
                "name": table2_name+ "split2",
                "header": header_2,
                "rows": rows_2
            }
        ]
    }
    
    return multi_table

def get_table_name(headers, table_num):
    """Generate a meaningful table name based on the headers"""
    if len(headers) > 1:
        # Use a descriptive non-join column if possible
        for h in headers[1:]:  # Skip the join column
            lower_h = h.lower()
            if any(keyword in lower_h for keyword in ['name', 'type', 'category', 'class', 'kind']):
                return f"{h.replace(' ', '_').lower()}_data"
    
    # Fallback to a generic name based on the first non-join column
    if len(headers) > 1:
        return f"{headers[1].replace(' ', '_').lower()}_info"
    
    # Last resort
    return f"table_{table_num}_data"

def process_dataset(split_name, output_dir):
    """Process a dataset split and save as multiple tables format"""
    dataset = load_dataset("wikitablequestions")[split_name]
    multi_table_dataset = []
    
    for idx, entry in enumerate(dataset):
        id_str = entry.get('id', f"{split_name}_{idx}")
        question = entry['question']
        answers = entry['answers']
        table = entry['table']
        
        # Convert to multi-table format
        multi_table_entry = split_table_vertically(table, question, answers, id_str)
        multi_table_dataset.append(multi_table_entry)
    
    # Save the dataset
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{split_name}_multi_table.json")
    
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(multi_table_dataset, file, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(multi_table_dataset)} examples from {split_name} dataset.")
    return output_path

def main():
    # Create output directory
    output_dir = "wikitablequestions_multi_table"
    os.makedirs(output_dir, exist_ok=True)
    
    output_files = {}
    for split in ['train', 'validation', 'test']:
        output_files[split] = process_dataset(split, output_dir)
    
    # Create a README file
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# WikiTableQuestions Multi-Table Dataset\n\n")
        f.write("This dataset is derived from the original WikiTableQuestions dataset, ")
        f.write("but with each table split into two related tables with a common key column.\n\n")
        f.write("## Format\n\n")
        f.write("Each entry contains:\n")
        f.write("- `id`: Unique identifier\n")
        f.write("- `question`: The original question\n")
        f.write("- `answers`: The original answers\n")
        f.write("- `tables`: An array of tables, each with:\n")
        f.write("  - `name`: A generated table name\n")
        f.write("  - `header`: Column headers\n")
        f.write("  - `rows`: Table data\n\n")
        f.write("## Files\n\n")
        for split, path in output_files.items():
            f.write(f"- {os.path.basename(path)}: {split} split\n")
    
    print(f"Dataset created in {output_dir}")
    print("The data structure is compatible with the multi-table operations in Chain-of-Tables")

if __name__ == "__main__":
    main()