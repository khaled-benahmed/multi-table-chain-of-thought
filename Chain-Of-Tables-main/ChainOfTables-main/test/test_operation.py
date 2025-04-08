# tests/test_operation.py
import pandas as pd
import pytest
from app.operation import (
    BeginOperation,
    Chain,
    EndOperation,
    SelectRow,
    SelectColumn,
)  # Adjust the import path according to your project structure


def test_select_row():
    data = {
        "Name": ["John", "Jane", "Alice", "Bob", "Charlie"],
        "Age": [28, 34, 29, 42, 24],
        "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
    }
    df = pd.DataFrame(data)

    sr = SelectRow()
    sr.args = ["1", "2"]

    expected_output = df.iloc[
        [1, 2]
    ]  # Assuming the expected output is the second row of df
    output = sr.perform(df)

    pd.testing.assert_frame_equal(output, expected_output)


def test_select_column():
    # Initialize the dataset
    data = {
        "Name": ["John", "Jane", "Alice", "Bob", "Charlie"],
        "Age": [28, 34, 29, 42, 24],
        "City": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
    }
    df = pd.DataFrame(data)

    # Initialize SelectColumn class instance and specify columns to select
    sc = SelectColumn()
    sc.args = ["Name", "City"]

    # Expected output: DataFrame with only 'Name' and 'City' columns
    expected_output = df[["Name", "City"]]

    # Perform the column selection
    output = sc.perform(df)

    # Assert that the output and expected output DataFrames are equal
    pd.testing.assert_frame_equal(output, expected_output)


def test_select_row_str():
    select_row = SelectRow()
    select_row.args = [1, 3]

    expected_str = "f_select_rows(1, 3)"
    assert (
        str(select_row) == expected_str
    ), "The __str__ method does not return the expected string representation."


def test_select_column_str():
    select_column = SelectColumn()
    select_column.args = ["Venue", "Crowd"]

    expected_str = "f_select_columns(Venue, Crowd)"
    assert (
        str(select_column) == expected_str
    ), "The __str__ method does not return the expected string representation."


def test_chain_str():

    begin_operation = BeginOperation()

    select_row = SelectRow()
    select_row.args = [1, 3]

    select_column = SelectColumn()
    select_column.args = ["Venue", "Crowd"]

    end_operation = EndOperation()

    chain = Chain(
        operations=[begin_operation, select_row, select_column, end_operation]
    )

    expected_str = "f_select_rows(1, 3) -> f_select_columns(Venue, Crowd) -> <END>"

    assert (
        str(chain) == expected_str
    ), "The __str__ method of the Chain does not return the expected string representation."
