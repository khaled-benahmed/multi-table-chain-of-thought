# WikiTableQuestions Multi-Table Dataset

This dataset is derived from the original WikiTableQuestions dataset, but with each table split into two related tables with a common key column.

## Format

Each entry contains:
- `id`: Unique identifier
- `question`: The original question
- `answers`: The original answers
- `tables`: An array of tables, each with:
  - `name`: A generated table name
  - `header`: Column headers
  - `rows`: Table data

## Files

- train_multi_table.json: train split
- validation_multi_table.json: validation split
- test_multi_table.json: test split
