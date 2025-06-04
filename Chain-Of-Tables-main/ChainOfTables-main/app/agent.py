import ast
import asyncio
import json
import random
import re
from collections import Counter
from io import StringIO

import pandas as pd

from config import config
from operation import (
    EndOperation,
    Operation,
    BeginOperation,
    Chain,
    SelectColumn,
    SelectRow,
    SelectTable,
    JoinTables
)
from prompt import PromptTemplate
from text_generation import AsyncClient  # Import here for each async task


class GuidanceStrategy:
    def build_grammar(self, agent, context, operation=None, tables=None):
        raise NotImplementedError()

class RegexGuidance(GuidanceStrategy):
    def generate_plan_pattern(self):
        plan_actions = [
            cls.name
            for cls in Operation.__subclasses__()
            if cls not in {BeginOperation, EndOperation}
        ]
        return r"(({})\([^)]*\) -> )*<END>".format(
            "|".join(re.escape(action) for action in plan_actions)
        )
    
    def build_grammar(self, agent, context, operation=None, tables=None):
        if context == "plan":
            pattern = self.generate_plan_pattern()
            return {"type": "regex", "value": pattern}
        elif context == "args" and operation and tables is not None:
            # Handle multi-table or single-table case
            if isinstance(tables, dict):
                pattern = operation.get_allowed_args_pattern(tables)
            else:
                pattern = operation.get_allowed_args_pattern(tables)
            return {"type": "regex", "value": pattern} if pattern else None
        elif context == "query":
            pattern = r"""\[\s*(?:(?:"[^"]*"|'[^']*'|\d+(?:\.\d+)?)(?:\s*,\s*(?:"[^"]*"|'[^']*'|\d+(?:\.\d+)?))*)?\s*\]"""
            return {"type": "regex", "value": pattern}
        
        return None

class NoneGuidance(GuidanceStrategy):
    def build_grammar(self, agent, context, operation=None, tables=None):
        return None

class JSONGuidance(GuidanceStrategy):
    def build_grammar(self, agent, context, operation=None, tables=None):
        if context == "plan":
            plan_actions = [
                cls.name
                for cls in Operation.__subclasses__()
                if cls not in {BeginOperation, EndOperation}
            ]
            schema = {
                "type": "object",
                "properties": {
                    "chain": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"enum": plan_actions},
                                "args": {"type": "object"}
                            },
                            "required": ["action", "args"]
                        }
                    }
                },
                "required": ["chain"]
            }
            return {"type": "json", "value": schema}

        elif context == "args" and operation and tables is not None:
            if hasattr(operation, "get_json_schema"):
                schema = operation.get_json_schema(tables)
                if schema:
                    return {"type": "json", "value": schema}
            return None

        elif context == "query":
            schema = {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
            return {"type": "json", "value": schema}

        return None


class GuidanceFactory:
    @staticmethod
    def get_strategy():
        guidance_type = config.get("guidance_type", "none").lower()
        if guidance_type == "regex":
            return RegexGuidance()
        elif guidance_type == "json":
            return JSONGuidance()
        elif guidance_type == "none":
            return NoneGuidance()
        else:
            raise ValueError(f"Unknown guidance_type: {guidance_type}")

        
class Agent:

    def __init__(self, logger):
        self.client = AsyncClient(config.get("inference_server_url"), timeout=None)
        self.logger = logger
        self.guidance_strategy = GuidanceFactory.get_strategy()
        self.actions = {cls.name: cls for cls in Operation.__subclasses__()}    

    def get_action(self, action):
        return self.actions[action]

    def get_actions_descritption(self):
        ignored_operations = {BeginOperation, EndOperation}  # Add any other operations to ignore here
        actions_description = "\n\n".join(
            f"{i+1}. {a().documentation()}"
            for i, a in enumerate(
                a for a in Operation.__subclasses__() if a not in ignored_operations
            )
        )
        return actions_description
        
    def get_possible_next_operations_prompt(self, chain):
        if config.get("use_possible_next_operations"):
            possible_operation_name = [possible_operation.name for possible_operation in chain.get_possible_next_operation()]
            if config.get("shuffle_possible_next_operations"):
                random.shuffle(possible_operation_name)
            possible_next_operations = ' or '.join(possible_operation_name) if possible_operation_name else '<END>'
            possible_next_operations_prompt = f"\nThe next operation must be one of {possible_next_operations}."
        else:
            possible_next_operations_prompt = ""
            
        return possible_next_operations_prompt
            
    def parse_dataframe_from_representation(self, table_representation):
        """
        Safely parse a DataFrame from string representation.
        
        Args:
            table_representation: String representation of tables
            
        Returns:
            Either a single DataFrame or a dictionary of DataFrames depending on the format
        """
        try:
            table_data = json.loads(table_representation)
            
            # Handle multi-table format
            if "tables" in table_data:
                result = {}
                for table_info in table_data["tables"]:
                    table_name = table_info.get("table_name", "")
                    columns = table_info.get("columns", [])
                    
                    # Extract data from table_column_priority
                    column_data = {}
                    for col_data in table_info.get("table_column_priority", []):
                        if len(col_data) > 0:
                            col_name = col_data[0]
                            if col_name in columns:
                                # Skip the first element which is the column name
                                column_data[col_name] = col_data[1:]
                    
                    # Ensure all columns have the same length
                    max_len = 0
                    for col, values in column_data.items():
                        max_len = max(max_len, len(values))
                    
                    # If no data found, set max_len to 0 for empty DataFrame
                    if max_len == 0:
                        max_len = 0
                    
                    # Create data dictionary with proper ordering using columns list
                    data = {}
                    for col in columns:
                        if col in column_data:
                            values = column_data[col][:]  # Create a copy
                            # Pad with empty strings if needed
                            if len(values) < max_len:
                                values.extend([""] * (max_len - len(values)))
                            data[col] = values
                        else:
                            data[col] = [""] * max_len
                    
                    # Create DataFrame using ordered columns to avoid ambiguity
                    if data and max_len > 0:
                        df = pd.DataFrame(data, columns=columns)
                    else:
                        # Create empty DataFrame with correct columns
                        df = pd.DataFrame(columns=columns)
                    
                    result[table_name] = df
                return result
            
            # Handle single table format
            else:
                columns = table_data.get("columns", [])
                
                # Extract data from table_column_priority
                column_data = {}
                for col_data in table_data.get("table_column_priority", []):
                    if len(col_data) > 0:
                        col_name = col_data[0]
                        if col_name in columns:
                            # Skip the first element which is the column name
                            column_data[col_name] = col_data[1:]
                
                # Ensure all columns have the same length
                max_len = 0
                for col, values in column_data.items():
                    max_len = max(max_len, len(values))
                
                # If no data found, set max_len to 0 for empty DataFrame
                if max_len == 0:
                    max_len = 0
                
                # Create data dictionary with proper ordering using columns list
                data = {}
                for col in columns:
                    if col in column_data:
                        values = column_data[col][:]  # Create a copy
                        # Pad with empty strings if needed
                        if len(values) < max_len:
                            values.extend([""] * (max_len - len(values)))
                        data[col] = values
                    else:
                        data[col] = [""] * max_len
                
                # Create DataFrame using ordered columns to avoid ambiguity
                if data and max_len > 0:
                    return pd.DataFrame(data, columns=columns)
                else:
                    # Create empty DataFrame with correct columns
                    return pd.DataFrame(columns=columns)
                
        except Exception as e:
            self.logger.warning(f"Error parsing DataFrame: {e}")
            try:
                # Try to determine if it was multi-table format for proper fallback
                fallback_data = json.loads(table_representation)
                return {} if "tables" in fallback_data else pd.DataFrame()
            except:
                return pd.DataFrame()

    async def dynamic_plan(self, tables_representation, question, chain, metadata):
        actions_description = self.get_actions_descritption()

        possible_next_operations_prompt = self.get_possible_next_operations_prompt(chain)
        
        template_name = "dynamic_plan_json" if config.get("guidance_type", "").lower() == "json" else "dynamic_plan"
        
        prompt_template = PromptTemplate.load(template_name)
        prompt = prompt_template.build(
            operations_instruction=actions_description,
            operation_chain_demo=Chain().documenation(),
            table=tables_representation,
            question=question,
            possible_next_operations_prompt=possible_next_operations_prompt,
            chain=chain,
        )

        # Get available tables for grammar generation
        tables = {}
        try:
            if isinstance(tables_representation, str):
                # Use the safe parsing method
                parsed_tables = self.parse_dataframe_from_representation(tables_representation)
                if isinstance(parsed_tables, dict):
                    tables = parsed_tables
                else:
                    tables = {"main_table": parsed_tables}
        except Exception as e:
            self.logger.warning(f"Error parsing tables for grammar generation: {e}")
            tables = {"main_table": pd.DataFrame()}
        
        grammar = self.guidance_strategy.build_grammar(agent=self, context="plan")
        
        response = await self.client.generate(
            prompt,
            max_new_tokens=100,
            stop_sequences=["<END>"],
            grammar=grammar,
        )

        response_text = response.generated_text
        metadata["generated_tokens"] += response.details.generated_tokens

        self.logger.debug(f"LLM Response: {response_text}")

        if config.get("guidance_type", "").lower() == "json":
            try:
                # Expecting a JSON object with a "chain" key.
                response_json = json.loads(response_text)
                op = response_json["chain"][0]
                action = op["action"]
                return self.get_action(action)
            except Exception as e:
                self.logger.warning(f"JSON parsing error: {e}")
                return EndOperation
        else:
            plan_actions = response_text.split("->")
            for text_action in plan_actions:
                if text_action.strip() == "":
                    continue
                match = re.search(r"(\w+)\(", text_action)
                action = match.group(1) if match else "No answer found"
                try:
                    return self.get_action(action)
                except KeyError:
                    pass
            return EndOperation

    def create_tables_representation(self, tables):
        """
        Create a string representation of tables for prompting.
        
        Args:
            tables: Either a DataFrame or a dictionary of table_name -> DataFrame
            
        Returns:
            A string representation of the tables
        """
        # Handle single table case
        if isinstance(tables, pd.DataFrame):
            # Use the original method for backward compatibility
            return self.create_single_table_representation(tables)
        
        # Handle multiple tables case
        representations = []
        
        for table_name, df in tables.items():
            # Create representation for each table - safely convert all values to strings
            table_column_priority = []
            for col in df.columns:
                # Convert column values to strings safely
                values = []
                for val in df[col].fillna("").values:
                    if isinstance(val, (int, float)):
                        values.append(str(val))
                    elif isinstance(val, str):
                        values.append(val)
                    else:
                        values.append(str(val))
                table_column_priority.append([col] + values)
            
            # Add table name and representation
            rep = {
                "table_name": table_name,
                "columns": list(df.columns),
                "table_column_priority": table_column_priority,
            }
            
            representations.append(rep)
        
        # Create final representation with all tables
        full_representation = {
            "tables": representations,
        }
        
        # Use compact separators and avoid unnecessary whitespace
        return json.dumps(full_representation, ensure_ascii=False, separators=(',', ':'))

    def create_single_table_representation(self, df):
        """Original method for single table representation with safety improvements"""
        # Transpose the table for `table_column_priority` format - safely convert all values to strings
        table_column_priority = []
        for col in df.columns:
            # Convert column values to strings safely
            values = []
            for val in df[col].fillna("").values:
                if isinstance(val, (int, float)):
                    values.append(str(val))
                elif isinstance(val, str):
                    values.append(val)
                else:
                    values.append(str(val))
            table_column_priority.append([col] + values)
        
        # Create the representation dictionary
        representation = {
            "columns": list(df.columns),
            "table_column_priority": table_column_priority,
        }
        
        # Use compact separators and avoid unnecessary whitespace
        return json.dumps(representation, ensure_ascii=False, separators=(',', ':'))

    async def generate_args(self, tables_representation, question, f, metadata):
        """
        Generate arguments for an operation.
        
        Args:
            tables_representation: String representation of tables
            question: The question to answer
            f: The operation to generate arguments for
            metadata: Metadata dictionary to update
            
        Returns:
            Arguments for the operation
        """
        # Select appropriate template based on operation type
        if isinstance(f, SelectTable):
            template_name = "generate_args_f_select_table"
        elif isinstance(f, JoinTables):
            template_name = "generate_args_f_join_tables"
        else:
            # Use original templates for existing operations
            template_name = f"generate_args_{f.name}"
            
        # Add JSON suffix if needed
        if config.get("guidance_type", "").lower() == "json":
            template_name += "_json"
        
        prompt_template = PromptTemplate.load(template_name)
        
        # Parse tables for grammar validation - use safe parsing method
        tables = {}
        try:
            if isinstance(tables_representation, str):
                parsed_tables = self.parse_dataframe_from_representation(tables_representation)
                if isinstance(parsed_tables, dict):
                    tables = parsed_tables
                else:
                    tables = {"main_table": parsed_tables}
        except Exception as e:
            self.logger.warning(f"Error parsing tables for grammar validation: {e}")
            tables = {"main_table": pd.DataFrame()}
                
        # For SelectColumn type operations, single table is still expected
        modified_table_rep = tables_representation
        if isinstance(f, SelectColumn) and isinstance(tables, dict) and "main_table" in tables:
            modified_table_rep = self.create_single_table_representation(tables["main_table"])
        
        prompt = prompt_template.build(
            operations=f.documentation(), 
            table=modified_table_rep,
            available_tables=tables_representation,  # For multi-table templates
            question=question
        )

        allowed_args_grammar = self.guidance_strategy.build_grammar(
            agent=self,
            context="args",
            operation=f,
            tables=tables
        )

        n_samples = f.generation_params.get("n_samples", 1)
        do_sample = f.generation_params.get("do_sample", False)
        temperature = f.generation_params.get("temperature", 0.001)
        
        tasks = [
            self.client.generate(
                prompt,
                max_new_tokens=250,
                grammar=allowed_args_grammar,
                do_sample=do_sample,
                temperature=temperature
            )
            for _ in range(n_samples)
        ]
        
        responses = await asyncio.gather(*tasks)
        generated_texts = []
        total_generated_tokens = 0
        for response in responses:
            generated_texts.append(response.generated_text)
            total_generated_tokens += response.details.generated_tokens

        metadata["generated_tokens"] += total_generated_tokens

        if config.get("guidance_type", "").lower() == "json":
            parsed_args = []
            for text in generated_texts:
                try:
                    parsed = json.loads(text)
                    parsed_args.append(parsed)
                except Exception as e:
                    self.logger.warning(f"Failed to parse JSON args: {e}")
            args = parsed_args[0] if parsed_args else {}
        else:
            # Extract arguments using regex - different patterns for different operations
            if isinstance(f, SelectTable):
                matches = [re.search(r'f_select_table\(\[(.*?)\]\)', x) for x in generated_texts]
                args = self.parse_table_selection_args(matches)
            elif isinstance(f, JoinTables):
                matches = [re.search(r'f_join_tables\(\'(.*?)\', \'(.*?)\', \'(.*?)\', \'(.*?)\'\)', x) for x in generated_texts]
                args = self.parse_join_tables_args(matches)
            else:
                # Original pattern for other operations
                matches = [m.group(1).strip() if (m := re.search(r'\((.*?)\)', x)) else None for x in generated_texts]
                response_counter = Counter(matches)
                try:
                    response_text, _ = response_counter.most_common(1)[0]
                    args = ast.literal_eval(response_text) if response_text is not None else "No answer found"
                except IndexError:
                    self.logger.warning("No valid arguments found in regex matches")
                    args = "No answer found"
                
        return args
    
    def parse_table_selection_args(self, matches):
        """Parse arguments for f_select_table operation"""
        valid_matches = [m.group(1) for m in matches if m]
        if not valid_matches:
            return []
            
        # Count and get most common pattern
        counter = Counter(valid_matches)
        try:
            most_common, _ = counter.most_common(1)[0]
                
            # Parse table names
            tables = [t.strip().strip("'\"") for t in most_common.split(",")]
            return tables
        except IndexError:
            self.logger.warning("No valid table selection arguments found")
            return []
        

        
    def parse_join_tables_args(self, matches):
        """Parse arguments for f_join_tables operation"""
        valid_matches = [(m.group(1), m.group(2), m.group(3), m.group(4)) for m in matches if m]
        if not valid_matches:
            return []
            
        try:
            # Get most common pattern
            counter = Counter(valid_matches)
            args, _ = counter.most_common(1)[0]
            return list(args)
        except IndexError:
            self.logger.warning("No valid join tables arguments found")
            return []

    async def direct_query(self, tables_representation, question, metadata):
        """
        Generate a direct answer to the question based on the tables.
        
        Args:
            tables_representation: String representation of tables
            question: The question to answer
            metadata: Metadata dictionary to update
            
        Returns:
            The answer to the question
        """
        template_name = "direct_query_json" if config.get("guidance_type", "").lower() == "json" else "direct_query"
        prompt_template = PromptTemplate.load(template_name)
        prompt = prompt_template.build(
            table_representation=tables_representation, question=question
        )

        # Parse tables for grammar validation - use safe parsing method
        tables = {}
        try:
            if isinstance(tables_representation, str):
                parsed_tables = self.parse_dataframe_from_representation(tables_representation)
                if isinstance(parsed_tables, dict):
                    tables = parsed_tables
                else:
                    tables = {"main_table": parsed_tables}
        except Exception as e:
            self.logger.warning(f"Error parsing tables for grammar validation: {e}")
            tables = {"main_table": pd.DataFrame()}
        
        grammar = self.guidance_strategy.build_grammar(
            agent=self,
            context="query",
            tables=tables
        )
        
        response = await self.client.generate(
            prompt,
            max_new_tokens=100,
            stop_sequences=["]"],
            grammar=grammar,
        )
        response_text = response.generated_text
        metadata["generated_tokens"] += response.details.generated_tokens
        return response_text


    async def chain_of_table(self, tables, question):
        """
        Process a question using chain of operations on tables.
        
        Args:
            tables: Either a single DataFrame or a dictionary of {table_name: DataFrame}
            question: The question to answer
            
        Returns:
            The answer, the operation chain, and metadata
        """
        # Convert single DataFrame to dictionary if needed
        if isinstance(tables, pd.DataFrame):
            tables = {"main_table": tables.copy()}
        else:
            # Ensure all values are DataFrames
            tables = {name: df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(df) 
                     for name, df in tables.items()}
            
        f = BeginOperation()
        chain = Chain([f])
        fail = 0
        metadata = {"reached_endoperation": False, "generated_tokens": 0}

        while not isinstance(f, EndOperation) and fail < config.get("MAX_FAILED_REPETITIONS") and chain.length() < config.get("MAX_CHAIN_LENGTH"):
            try:
                # Create representation of current tables state
                tables_representation = await asyncio.to_thread(self.create_tables_representation, tables)

                f, args = None, None

                f_class = await self.dynamic_plan(
                    tables_representation, question, chain, metadata
                )
                f = f_class()
                if isinstance(f, EndOperation):
                    # If an end operation is reached within the chain
                    # so the LLM decided to stop processing, and not an artifact of failed planning
                    metadata["reached_endoperation"] = True
                    break

                args = await self.generate_args(
                    tables_representation,
                    question,
                    f,
                    metadata,
                )
                f.args = args

                # Execute operation on tables
                tables = await asyncio.to_thread(f.perform, tables)
                chain.operations.append(f)
                
            except Exception as e:
                self.logger.warning(
                    (
                        f"Planning or operation execution failed. \n"
                        f"Intended function: {f}\nArgs: {args}\nTables:\n"
                        f"{tables_representation if 'tables_representation' in locals() else 'Table representation not available'}\nError: {e}"
                    )
                )
                fail += 1
                
        # For the final query, use the main table if it's a single table result
        try:
            if len(tables) == 1:
                main_table_name = next(iter(tables.keys()))
                table_representation = await asyncio.to_thread(self.create_single_table_representation, tables[main_table_name])
            else:
                table_representation = await asyncio.to_thread(self.create_tables_representation, tables)
                
            answer = None
            try:
                answer = await self.direct_query(table_representation, question, metadata)
            except Exception as e:
                self.logger.warning("Failed to generate an answer because of the following error:\n" + str(e))
                
            return answer, chain, metadata
        except Exception as e:
            self.logger.warning(f"Error creating final table representation: {e}")
            return "Failed to generate answer due to an error", chain, metadata