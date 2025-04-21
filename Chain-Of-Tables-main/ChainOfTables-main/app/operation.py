from typing import List, Dict, Optional, Union
import pandas as pd
from dataclasses import dataclass, field
import re
import json
import numpy as np
from collections import defaultdict


class Operation:
    name = "generic_operation"
    args = []
    description = "Generic operation on a DataFrame"

    generation_params = {"do_sample": False}
    
    def __init__(self, args=[]):
        self.args = args

    def perform(self, dataframe):
        raise NotImplementedError("Subclasses must implement this method.")

    def documentation(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def get_allowed_args_pattern(self, dataframe):
        return None

    def __str__(self):
        if isinstance(self.args, list):
            args_str = ", ".join(map(str, self.args))
        else:
            args_str = str(self.args)
        return f"{self.name}({args_str})"


@dataclass
class OperationExample:
    data: dict
    question: str
    function: str
    explaination: str

    def __str__(self) -> str:
        return (
            f"For example,\n/*\n{pd.DataFrame(self.data).to_csv()}*/\n"
            f"Question : {self.question}\nFunction: {self.function}\n"
            f"Explanation: {self.explaination}"
        )


class SelectColumn(Operation):
    name = "f_select_column"
    description = (
        "Retrieves the attributes (columns) specified from the relation (table)."
    )

    generation_params = {"do_sample": True, "temperature": 1.0, "n_samples": 8}

    def perform(self, dataframe):
        return dataframe[self.args]

    def get_allowed_args_pattern(self, dataframe):
        columns_pattern = "|".join(
            f'"{re.escape(column)}"' for column in dataframe.columns.tolist()
        )
        regex_pattern = rf"Explanation: .*\nAnswer: f_select_column\(\[({columns_pattern})(, ({columns_pattern}))*\]\)"
        return regex_pattern

    def documentation(self):
        instruction = (
            "If the table only needs a few columns to answer the question, "
            "we use f_select_column() to select these columns for it."
        )

        example = OperationExample(
            data={
                "Home Team": ["St Kilda", "South Melbourne", "Richmond"],
                "Home Team Score": ["13.12 (90)", "9.12 (66)", "20.17 (137)"],
                "Away Team": ["Melbourne", "Footscray", "Fitzroy"],
                "Away Team Score": ["13.11 (89)", "11.13 (79)", "13.22 (100)"],
                "Venue": ["Moorabbin Oval", "Lake Oval", "MCG"],
                "Crowd": [18836, 9154, 27651],
            },
            question="What are the venues and crowds for each game?",
            function="f_select_column('Venue', 'Crowd')",
            explaination=(
                "The question asks for information about the venues and the crowds for each game. "
                "To answer this, we need to select the data from the 'Venue' and 'Crowd' columns. "
                "We use f_select_column('Venue', 'Crowd') to select these specific columns."
            ),
        )

        return f"{instruction} {example}"


    def get_json_schema(self, dataframe):
        # Allowed columns are those present in the DataFrame.
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(dataframe.columns)}
                }
            },
            "required": ["columns"]
        }

class SelectRow(Operation):
    name = "f_select_row"

    generation_params = {"do_sample": True, "temperature": 1.0, "n_samples": 8}

    def perform(self, dataframe):
        row_nr = [i - 1 for i in self.args]
        return dataframe.iloc[row_nr]

    def get_allowed_args_pattern(self, dataframe):
        number_parts = [str(i) for i in range(1, len(dataframe) + 1)]
        number_pattern = "|".join(number_parts)
        # Create the full regex pattern
        regex_pattern = rf"Explanation: .*\nAnswer: f_select_row\(\[({number_pattern})(, ({number_pattern}))*\]\)"
        return regex_pattern

    def documentation(self):
        instruction = (
            "If the table only needs a few rows to answer the question,"
            "we use f_select_row() to select these rows for it."
        )

        example = OperationExample(
            data={
                "Home Team": ["St Kilda", "South Melbourne", "Richmond"],
                "Home Team Score": ["13.12 (90)", "9.12 (66)", "20.17 (137)"],
                "Away Team": ["Melbourne", "Footscray", "Fitzroy"],
                "Away Team Score": ["13.11 (89)", "11.13 (79)", "13.22 (100)"],
                "Venue": ["Moorabbin Oval", "Lake Oval", "MCG"],
                "Crowd": [18836, 9154, 27651],
            },
            question="Whose home team score is higher, richmond or st kilda?",
            function="f_select_row(row 0, row 2)",
            explaination=(
                "The question asks about the home team score of richmond and st kilda."
                "We need to know the the information of richmond and st kilda in row 0 and row 2. "
                "We select row 0 and row 2."
            ),
        )

        return f"{instruction} {example}"


    def get_json_schema(self, dataframe):
        return {
            "type": "object",
            "properties": {
                "rows": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": len(dataframe) - 1
                    }
                }
            },
            "required": ["rows"]
        }

class BeginOperation(Operation):
    def __str__(self):
        return ""

    def documentation(self):
        return "begin"


class EndOperation(Operation):
    name = "<END>"

    def __str__(self):
        return "<END>"

    def documentation(self):
        return "Terminal action of the plan. Each plan must contain this action as its final action."


class AddColumn(Operation):
    name = "f_add_column"

    def documentation(self):
        instruction = (
            "If the table does not have the needed column to answer the question, "
            "we use f_add_column() to add a new column for it."
        )

        example = OperationExample(
            data = {
                "rank": ["", "", ""],
                "lane": [5, 6, 3],
                "player": ["olga tereshkova (kaz)", "manjeet kaur (ind)", "asami tanno (jpn)"],
                "time": [51.86, 52.17, 53.04],
            },
            question="how many athletes are from japan?",
            function="f_add_column('country of athlete')",
            explaination=(
                "The question is about the number of athletes from japan. We need to known the country of each athlete. "
                "There is no column of the country of athletes. We add a column 'country of athlete.'"
            ),
        )

        return f"{instruction} {example}"

    def get_allowed_args_pattern(self, dataframe):
        regex_pattern = (
            rf'Explanation: .*\nAnswer: f_add_column\(\["[^"]+", "(\s*\d+ \| .+? \\n)"'
            rf'{{{len(dataframe) - 1}}}(\s*\d+ \| [^|]+?)"\]\)'
        )
        return regex_pattern

    def perform(self, dataframe):
        row_name = self.args[0]
        rows = self.args[1].split("\n")
        data = [row.split("|") for row in rows]
        added_df = pd.DataFrame(data, columns=["ID", row_name])
        dataframe[row_name] = added_df[row_name]
        return dataframe

    def get_json_schema(self, dataframe):
        # For AddColumn, we expect a column name and a list of values (one per row).
        return {
            "type": "object",
            "properties": {
                "column_name": {"type": "string"},
                "values": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": len(dataframe),
                    "maxItems": len(dataframe)
                }
            },
            "required": ["column_name", "values"]
        }

class SortBy(Operation):
    name = "f_sort_by"

    def documentation(self):
        instruction = "If the question asks about the order of items in a column, we use f_sort_by() to sort the items."

        example = OperationExample(
            data = {
                "position": [1, 10, 3],
                "club": ["malaga cf", "cp merida", "cd numancia"],
                "played": [42, 42, 42],
                "points": [79, 59, 73],
            },
            question="in which position did cd numancia finish?",
            function="f_sort_column('position')",
            explaination=(
                "The question is about cd numancia position. We need to know the order of position from last to front. " 
                "We sort the rows according to column 'position'."
            ),
        )

        return f"{instruction} {example}"

    def get_allowed_args_pattern(self, dataframe):
        columns_pattern = "|".join(
            f'"{re.escape(column)}"' for column in dataframe.columns.tolist()
        )
        regex_pattern = (
            rf"Explanation: .*\nAnswer: f_sort_by\(\[\[({columns_pattern})(, ({columns_pattern}))*\], "
            rf"'(ascending|descending)'\]\)"
        )
        return regex_pattern

    def perform(self, dataframe):
        by = self.args[0]
        ascending = True if self.args[1] == "ascending" else False
        return dataframe.sort_values(by, ascending=ascending)

    def get_json_schema(self, dataframe):
        return {
            "type": "object",
            "properties": {
                "column": {"type": "string", "enum": list(dataframe.columns)},
                "order": {"type": "string", "enum": ["ascending", "descending"]}
            },
            "required": ["column", "order"]
        }


class GroupBy(Operation):
    name = "f_group_by"

    def documentation(self):
        instruction = (            
            "If the question asks about items with the same value and the number"
            "of these items, we use f_group_by() to group the items."
        )

        example = OperationExample(
            data = {
                "district": ["district 1", "district 1", "district 2"],
                "name": ["nelson albano", "robert andrzejczak", "john f. amodeo"],
                "party": ["dem", "dem", "rep"],
                "residence": ["vineland", "middle twp.", "margate"],
                "first served": ["2006", "2013†", "2008"],
            },
            question="How many districts are democratic?",
            function="f_group_column('party')",
            explaination=(
                "The question wants to count democratic districts. We need to know the number of them in the table. "
                "We group the rows according to column 'party'."
            ),
        )

        return f"{instruction} {example}"
    
    def get_allowed_args_pattern(self, dataframe):
        columns_pattern = "|".join(
            f'"{re.escape(column)}"' for column in dataframe.columns.tolist()
        )
        regex_pattern = rf"Explanation: .*\nAnswer: f_group_by\(\[({columns_pattern})(, ({columns_pattern}))*\]\)"
        return regex_pattern

    def perform(self, dataframe):
        result = dataframe.groupby(by=self.args).count()
        return result

    def get_json_schema(self, dataframe):
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(dataframe.columns)}
                }
            },
            "required": ["columns"]
        }

# Multi-Table Operations

class SelectTable(Operation):
    """Operation to select one or more tables from a set of tables based on relevance to a question."""
    name = "f_select_table"
    description = "Selects the most relevant tables for answering a question."
    
    generation_params = {"do_sample": True, "temperature": 0.7, "n_samples": 5}
    
    def perform(self, tables_dict):
        """
        Select specific tables from the tables_dict.
        
        Args:
            tables_dict: Dictionary of table_name -> DataFrame
            
        Returns:
            Dictionary containing only the selected tables
        """
        selected_tables = {}
        for table_name in self.args:
            if table_name in tables_dict:
                selected_tables[table_name] = tables_dict[table_name]
        return selected_tables
    
    def get_allowed_args_pattern(self, tables_dict):
        table_names_pattern = "|".join(
            f'"{re.escape(name)}"' for name in tables_dict.keys()
        )
        regex_pattern = rf"Explanation: .*\nAnswer: f_select_table\(\[({table_names_pattern})(, ({table_names_pattern}))*\]\)"
        return regex_pattern
    
    def documentation(self):
        instruction = (
            "When working with multiple tables, we use f_select_table() to identify and select "
            "the tables that are most relevant to answering the question."
        )
        
        example = OperationExample(
            data={
                "table_names": ["population_data", "gdp_data", "export_data"],
                "population_data": pd.DataFrame({
                    "Country": ["USA", "Canada", "Japan"],
                    "Population": [331, 38, 125]
                }).to_dict(),
                "gdp_data": pd.DataFrame({
                    "Country": ["USA", "Canada", "Japan", "Germany"],
                    "GDP": [21000, 1800, 5000, 4000]
                }).to_dict(),
                "export_data": pd.DataFrame({
                    "Country": ["USA", "Canada", "China"],
                    "Exports": [2500, 550, 2700]
                }).to_dict()
            },
            question="What is the GDP per capita of Japan?",
            function="f_select_table(['population_data', 'gdp_data'])",
            explaination=(
                "The question asks about GDP per capita, which requires both population data and GDP data. "
                "We need to select both the population_data table and the gdp_data table. "
                "The export_data table is not needed for this question."
            )
        )
        
        return f"{instruction} {example}"
    
    def get_json_schema(self, tables_dict):
        return {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(tables_dict.keys())}
                }
            },
            "required": ["tables"]
        }


class NormalizeColumn(Operation):
    """Operation to normalize column names across tables to align them."""
    name = "f_normalize_column"
    description = "Normalizes column names across tables to align similar columns."
    
    generation_params = {"do_sample": True, "temperature": 0.5, "n_samples": 5}
    
    def perform(self, tables_dict):
        """
        Normalize specified columns across tables.
        
        Args:
            tables_dict: Dictionary of table_name -> DataFrame
            
        Returns:
            Updated dictionary with normalized columns
        """
        if not self.args or len(self.args) < 2:
            return tables_dict
        
        mappings = {}
        target_name = self.args[0]
        
        # Parse the mappings from args
        # Format: [target_name, [table1, col1], [table2, col2], ...]
        for mapping in self.args[1:]:
            if isinstance(mapping, list) and len(mapping) == 2:
                table_name, column_name = mapping
                mappings[table_name] = column_name
        
        # Rename columns in tables
        updated_tables = {}
        for table_name, df in tables_dict.items():
            if table_name in mappings:
                col_name = mappings[table_name]
                if col_name in df.columns:
                    updated_df = df.copy()
                    updated_df = updated_df.rename(columns={col_name: target_name})
                    updated_tables[table_name] = updated_df
                else:
                    updated_tables[table_name] = df
            else:
                updated_tables[table_name] = df
                
        return updated_tables
    
    def documentation(self):
        instruction = (
            "When working with multiple tables that have different column names representing the same concept, "
            "we use f_normalize_column() to standardize these columns to a common name."
        )
        
        example = OperationExample(
            data={
                "table_names": ["population_table", "economic_table"],
                "population_table": pd.DataFrame({
                    "Nation": ["USA", "Canada", "Japan"],
                    "Population": [331, 38, 125]
                }).to_dict(),
                "economic_table": pd.DataFrame({
                    "Country Name": ["USA", "Canada", "Japan", "Germany"],
                    "GDP": [21000, 1800, 5000, 4000]
                }).to_dict()
            },
            question="Compare the GDP and population of Japan.",
            function="f_normalize_column('Country', ['population_table', 'Nation'], ['economic_table', 'Country Name'])",
            explaination=(
                "To join these tables for comparison, we need to normalize the country name columns. "
                "In population_table, the column is called 'Nation', and in economic_table, it's called 'Country Name'. "
                "We standardize both to the common name 'Country'."
            )
        )
        
        return f"{instruction} {example}"
    
    def get_json_schema(self, tables_dict):
        properties = {
            "target_name": {"type": "string"},
            "mappings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string", "enum": list(tables_dict.keys())},
                        "column_name": {"type": "string"}
                    },
                    "required": ["table_name", "column_name"]
                }
            }
        }
        return {
            "type": "object",
            "properties": properties,
            "required": ["target_name", "mappings"]
        }


class JoinTables(Operation):
    """Operation to join two tables on a specified key."""
    name = "f_join_tables"
    description = "Joins two tables based on a common key column."
    
    generation_params = {"do_sample": True, "temperature": 0.5, "n_samples": 5}
    
    def perform(self, tables_dict):
        """
        Join two tables on a specified key.
        
        Args:
            tables_dict: Dictionary of table_name -> DataFrame
            
        Returns:
            Updated dictionary with the joined table added
        """
        if len(self.args) != 4:
            return tables_dict
        
        table1_name, table2_name, join_column, new_table_name = self.args
        
        if table1_name not in tables_dict or table2_name not in tables_dict:
            return tables_dict
        
        df1 = tables_dict[table1_name]
        df2 = tables_dict[table2_name]
        
        # Ensure join column exists in both tables
        if join_column not in df1.columns or join_column not in df2.columns:
            return tables_dict
        
        # Perform the join
        joined_df = pd.merge(df1, df2, on=join_column, how='inner')
        
        # Add joined table to the dictionary
        result = tables_dict.copy()
        result[new_table_name] = joined_df
        
        return result
    
    def documentation(self):
        instruction = (
            "When data needed to answer a question is split across multiple tables, "
            "we use f_join_tables() to combine them based on a common key."
        )
        
        example = OperationExample(
            data={
                "table_names": ["population_data", "gdp_data"],
                "population_data": pd.DataFrame({
                    "Country": ["USA", "Canada", "Japan"],
                    "Population": [331, 38, 125]
                }).to_dict(),
                "gdp_data": pd.DataFrame({
                    "Country": ["USA", "Canada", "Japan", "Germany"],
                    "GDP": [21000, 1800, 5000, 4000]
                }).to_dict()
            },
            question="What is the GDP per capita for each country?",
            function="f_join_tables('population_data', 'gdp_data', 'Country', 'combined_data')",
            explaination=(
                "To calculate GDP per capita, we need both population and GDP for each country. "
                "We join the population_data and gdp_data tables using the common 'Country' column. "
                "The result is stored in a new table named 'combined_data'."
            )
        )
        
        return f"{instruction} {example}"
    
    def get_json_schema(self, tables_dict):
        table_names = list(tables_dict.keys())
        
        column_names = set()
        for df in tables_dict.values():
            column_names.update(df.columns)
        
        return {
            "type": "object",
            "properties": {
                "table1": {"type": "string", "enum": table_names},
                "table2": {"type": "string", "enum": table_names},
                "join_column": {"type": "string", "enum": list(column_names)},
                "result_table_name": {"type": "string"}
            },
            "required": ["table1", "table2", "join_column", "result_table_name"]
        }


@dataclass
class Chain:
    operations: List[Operation] = field(default_factory=list)

    def __str__(self):
        filtered_operations = [
            o for o in self.operations if not isinstance(o, BeginOperation)
        ]
        return " -> ".join(str(o) for o in filtered_operations)
    
    def length(self):
        return len(self.operations)

    def documenation(self):
        # First scenario data - single table
        data1 = {
            "Date": ["2001/01/02", "2002/08/06", "2005/03/24"],
            "Division": [2, 2, 2],
            "League": ["USL A-League", "USL A-League", "USL First Division"],
            "Regular Season": ["4th, Western", "2nd, Pacific", "5th"],
            "Playoffs": ["Quarterfinals", "1st Round", "Quarterfinals"],
            "Open Cup": ["Did not qualify", "Did not qualify", "4th Round"],
        }
        question1 = (
            "what was the last year where this team was a part of the usl a-league?"
        )
        function_chain1 = Chain(
            [BeginOperation(),
                AddColumn(["Year"]),
                SelectRow(["1", "2"]),
                SelectColumn(["Year", "League"]),
                SortBy(["Year"]),
                EndOperation(),
            ]
        )

        example1 = OperationChainExample(
            data=data1, question=question1, function_chain=function_chain1
        )
        
        # Second scenario data - single table
        data2 = {
            "rank": [1, 2, 3],
            "lane": [6, 5, 4],
            "athlete": ["manjeet kaur (ind)", "olga tereshkova (kaz)", "pinki pramanik (ind)"],
            "time": [52.17, 51.86, 53.06]
        }
        question2 = "How many athletes are from India?"
        function_chain2 = Chain([
            BeginOperation(),
            AddColumn(["country of athletes"]),
            SelectRow(["1", "3"]),
            SelectColumn(["athlete", "country of athletes"]),
            GroupBy(["country of athletes"]),
            EndOperation(),
        ])
        example2 = OperationChainExample(data=data2, question=question2, function_chain=function_chain2)

        # Third scenario data - Multi-table operations example
        table1_data = {
            "Country": ["USA", "Canada", "Japan"],
            "Population": [331, 38, 125]
        }
        
        table2_data = {
            "Nation": ["USA", "Canada", "Japan", "Germany"],
            "GDP": [21000, 1800, 5000, 4000]
        }
        
        question3 = "What is the GDP per capita of Japan?"
        function_chain3 = Chain([
            BeginOperation(),
            SelectTable(["population_data", "gdp_data"]),
            NormalizeColumn(["Country", ["gdp_data", "Nation"]]),
            JoinTables(["population_data", "gdp_data", "Country", "combined_data"]),
            SelectColumn(["Country", "Population", "GDP"]),
            EndOperation(),
        ])
        example3 = OperationChainExample(
            data={"population_data": table1_data, "gdp_data": table2_data}, 
            question=question3, 
            function_chain=function_chain3
        )

        # Fourth scenario data - Another multi-table example
        table3_data = {
            "Country": ["USA", "Canada", "Japan", "Germany", "France"],
            "Continent": ["North America", "North America", "Asia", "Europe", "Europe"],
        }
        
        table4_data = {
            "Country": ["USA", "Canada", "Japan", "Germany", "France"],
            "GDP": [21000, 1800, 5000, 4000, 2700]
        }
        
        question4 = "What is the GDP of European countries?"
        function_chain4 = Chain([
            BeginOperation(),
            SelectTable(["continent_data", "gdp_data"]),
            JoinTables(["continent_data", "gdp_data", "Country", "combined_data"]),
            SelectColumn(["Country", "Continent", "GDP"]),
            EndOperation(),
        ])
        example4 = OperationChainExample(
            data={"continent_data": table3_data, "gdp_data": table4_data}, 
            question=question4, 
            function_chain=function_chain4
        )

        return (
            f"{example1}\n"
            f"{example2}\n"
            f"{example3}\n"
            f"{example4}\n"
        )

    # Updated transition map to prioritize SelectTable as the first operation and 
    # remove AggregateTable and FilterTable
    possible_next_operation_dict = {
        BeginOperation: [
            SelectTable,  # For multi-table scenarios, start by selecting relevant tables
            AddColumn,
            SelectRow,
            SelectColumn,
            GroupBy,
            SortBy,
        ],
        SelectTable: [
            NormalizeColumn,  # After selecting tables, normalize columns if needed
            JoinTables,       # Or directly join if columns already aligned
            SelectColumn,     # Or select relevant columns from specific tables
            SelectRow,        # Or select specific rows
            EndOperation,     # If only table selection was needed
        ],
        NormalizeColumn: [
            JoinTables,      # After normalizing, typically join the tables
            SelectColumn,    # Or select specific columns
            SelectRow,       # Or select specific rows
            EndOperation,    # If normalization was the main operation needed
        ],
        JoinTables: [
            SelectColumn,    # Or select specific columns
            SelectRow,       # Or select specific rows
            AddColumn,       # Or add derived columns
            GroupBy,         # Or group by certain columns
            SortBy,          # Or sort the results
            EndOperation,    # If joining was the main operation
        ],
        AddColumn: [
            SelectRow,
            SelectColumn,
            GroupBy,
            SortBy,
            EndOperation,
        ],
        SelectRow: [
            SelectColumn,
            GroupBy,
            SortBy,
            EndOperation,
        ],
        SelectColumn: [
            GroupBy,
            SortBy,
            EndOperation,
        ],
        GroupBy: [
            SortBy,
            EndOperation,
        ],
        SortBy: [
            EndOperation,
        ],
    }

    def get_possible_next_operation(self):
        return Chain.possible_next_operation_dict[self.operations[-1].__class__]


@dataclass
class OperationChainExample:
    data: dict
    question: str
    function_chain: Chain

    def __str__(self) -> str:
        if isinstance(next(iter(self.data.values())), dict):
            # Multi-table scenario
            tables_str = ""
            for table_name, table_data in self.data.items():
                tables_str += f"Table {table_name}:\n{pd.DataFrame(table_data).to_csv()}\n"
            return (
                f"\\*\n{tables_str}*/\n"
                f"Question: {self.question}\n"
                f"Function Chain: {str(self.function_chain)}\n"
            )
        else:
            # Single table scenario
            return (
                f"\\*\n{pd.DataFrame(self.data).to_csv()}*/\n"
                f"Question: {self.question}\n"
                f"Function Chain: {str(self.function_chain)}\n"
            )