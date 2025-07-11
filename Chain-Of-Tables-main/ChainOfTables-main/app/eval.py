#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adjusted version of the Official Evaluator for WikiTableQuestions Dataset

Main changes:
- runs on python3
- uses the Evaluator class to evaluate
- uses question to search instead of id 


Official documentation:
There are 3 value types
1. String (str)
2. Number (float)
3. Date (a struct with 3 fields: year, month, and date)
   Some fields (but not all) can be left unspecified. However, if only the year
   is specified, the date is automatically converted into a number.

Target denotation = a set of items
- Each item T is a raw unicode string from Mechanical Turk
- If T can be converted to a number or date (via Stanford CoreNLP), the
    converted value (number T_N or date T_D) is precomputed

Predicted denotation = a set of items
- Each item P is a string, a number, or a date
- If P is read from a text file, assume the following
  - A string that can be converted into a number (float) is converted into a
    number
  - A string of the form "yyyy-mm-dd" is converted into a date. Unspecified
    fields can be marked as "xx". For example, "xx-01-02" represents the date
    January 2nd of an unknown year.
  - Otherwise, it is kept as a string

The predicted denotation is correct if
1. The sizes of the target denotation and the predicted denotation are equal
2. Each item in the target denotation matches an item in the predicted
    denotation

A target item T matches a predicted item P if one of the following is true:
1. normalize(raw string of T) and normalize(string form of P) are identical.
   The normalize method performs the following normalizations on strings:
   - Remove diacritics (é → e)
   - Convert smart quotes (‘’´`“”) and dashes (‐‑‒–—−) into ASCII ones
   - Remove citations (trailing •♦†‡*#+ or [...])
   - Remove details in parenthesis (trailing (...))
   - Remove outermost quotation marks
   - Remove trailing period (.)
   - Convert to lowercase
   - Collapse multiple whitespaces and strip outermost whitespaces
2. T can be interpreted as a number T_N, P is a number, and P = T_N
3. T can be interpreted as a date T_D, P is a date, and P = T_D
   (exact match on all fields; e.g., xx-01-12 and 1990-01-12 do not match)
"""
__version__ = '1.0.2'

import sys, os, re, argparse
import unicodedata
from math import isnan, isinf
from abc import ABC, abstractmethod

################ String Normalization ################

def normalize(x):
    if not isinstance(x, str):
        x = str(x, 'utf8', errors='ignore')
    # Remove diacritics
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub(r"[‘’´`]", "'", x)
    x = re.sub(r"[“”]", "\"", x)
    x = re.sub(r"[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub(r"((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub(r"(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub(r'^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub(r'\s+', ' ', x, flags=re.UNICODE).lower().strip()
    return x


################ Value Types ################

class Value(ABC):

    # Should be populated with the normalized string
    _normalized = None

    @abstractmethod
    def match(self, other):
        """Return True if the value matches the other value.

        Args:
            other (Value)
        Returns:
            a boolean
        """
        pass

    @property
    def normalized(self):
        return self._normalized


class StringValue(Value):

    def __init__(self, content):
        assert isinstance(content, str)
        self._normalized = normalize(content)
        self._hash = hash(self._normalized)

    def __eq__(self, other):
        return isinstance(other, StringValue) and self.normalized == other.normalized

    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'S' +  str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        return self.normalized == other.normalized


class NumberValue(Value):

    def __init__(self, amount, original_string=None):
        assert isinstance(amount, (int, float))
        if abs(amount - round(amount)) < 1e-6:
            self._amount = int(amount)
        else:
            self._amount = float(amount)
        if not original_string:
            self._normalized = str(self._amount)
        else:
            self._normalized = normalize(original_string)
        self._hash = hash(self._amount)

    @property
    def amount(self):
        return self._amount

    def __eq__(self, other):
        return isinstance(other, NumberValue) and self.amount == other.amount

    def __hash__(self):
        return self._hash

    def __str__(self):
        return ('N(%f)' % self.amount) + str([self.normalized])
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, NumberValue):
            return abs(self.amount - other.amount) < 1e-6
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a number.

        Return:
            the number (int or float) if successful; otherwise None.
        """
        try:
            return int(text)
        except:
            try:
                amount = float(text)
                assert not isnan(amount) and not isinf(amount)
                return amount
            except:
                return None


class DateValue(Value):

    def __init__(self, year, month, day, original_string=None):
        """Create a new DateValue. Placeholders are marked as -1."""
        assert isinstance(year, int)
        assert isinstance(month, int) and (month == -1 or 1 <= month <= 12)
        assert isinstance(day, int) and (day == -1 or 1 <= day <= 31)
        assert not (year == month == day == -1)
        self._year = year
        self._month = month
        self._day = day
        if not original_string:
            self._normalized = '{}-{}-{}'.format(
                year if year != -1 else 'xx',
                month if month != -1 else 'xx',
                day if day != '-1' else 'xx')
        else:
            self._normalized = normalize(original_string)
        self._hash = hash((self._year, self._month, self._day))

    @property
    def ymd(self):
        return (self._year, self._month, self._day)

    def __eq__(self, other):
        return isinstance(other, DateValue) and self.ymd == other.ymd

    def __hash__(self):
        return self._hash

    def __str__(self):
        return (('D(%d,%d,%d)' % (self._year, self._month, self._day))
                + str([self._normalized]))
    __repr__ = __str__

    def match(self, other):
        assert isinstance(other, Value)
        if self.normalized == other.normalized:
            return True
        if isinstance(other, DateValue):
            return self.ymd == other.ymd
        return False

    @staticmethod
    def parse(text):
        """Try to parse into a date.

        Return:
            tuple (year, month, date) if successful; otherwise None.
        """
        try:
            ymd = text.lower().split('-')
            assert len(ymd) == 3
            year = -1 if ymd[0] in ('xx', 'xxxx') else int(ymd[0])
            month = -1 if ymd[1] == 'xx' else int(ymd[1])
            day = -1 if ymd[2] == 'xx' else int(ymd[2])
            assert not (year == month == day == -1)
            assert month == -1 or 1 <= month <= 12
            assert day == -1 or 1 <= day <= 31
            return (year, month, day)
        except:
            return None


################ Value Instantiation ################

def to_value(original_string, corenlp_value=None):
    """Convert the string to Value object.

    Args:
        original_string (str): Original string
        corenlp_value (str): Optional value returned from CoreNLP
    Returns:
        Value
    """
    if isinstance(original_string, Value):
        # Already a Value
        return original_string
    if not corenlp_value:
        corenlp_value = original_string
    # Number?
    amount = NumberValue.parse(corenlp_value)
    if amount is not None:
        return NumberValue(amount, original_string)
    # Date?
    ymd = DateValue.parse(corenlp_value)
    if ymd is not None:
        if ymd[1] == ymd[2] == -1:
            return NumberValue(ymd[0], original_string)
        else:
            return DateValue(ymd[0], ymd[1], ymd[2], original_string)
    # String.
    return StringValue(original_string)

def to_value_list(original_strings, corenlp_values=None):
    """Convert a list of strings to a list of Values

    Args:
        original_strings (list[str])
        corenlp_values (list[str or None])
    Returns:
        list[Value]
    """
    assert isinstance(original_strings, (list, tuple, set))
    if corenlp_values is not None:
        assert isinstance(corenlp_values, (list, tuple, set))
        assert len(original_strings) == len(corenlp_values)
        return list(set(to_value(x, y) for (x, y)
                in zip(original_strings, corenlp_values)))
    else:
        return list(set(to_value(x) for x in original_strings))


################ Check the Predicted Denotations ################

def check_denotation(target_values, predicted_values):
    """Return True if the predicted denotation is correct.
    
    Args:
        target_values (list[Value])
        predicted_values (list[Value])
    Returns:
        bool
    """
    # Check size
    if len(target_values) != len(predicted_values):
        return False
    # Check items
    for target in target_values:
        if not any(target.match(pred) for pred in predicted_values):
            return False
    return True


################ Batch Mode ################

def tsv_unescape(x):
    """Unescape strings in the TSV file.
    Escaped characters include:
        newline (0x10) -> backslash + n
        vertical bar (0x7C) -> backslash + p
        backslash (0x5C) -> backslash + backslash

    Args:
        x (str)
    Returns:
        a str
    """
    return x.replace(r'\n', '\n').replace(r'\p', '|').replace('\\\\', '\\')

def tsv_unescape_list(x):
    """Unescape a list in the TSV file.
    List items are joined with vertical bars (0x5C)

    Args:
        x (str)
    Returns:
        a list of str
    """
    return [tsv_unescape(y) for y in x.split('|')]

class Evaluator:
    def __init__(self, tagged_dataset_path=os.path.join('.', 'tagged', 'data')):
        self.target_values_map = {}
        self.load_tagged_dataset(tagged_dataset_path)
    
    def load_tagged_dataset(self, tagged_dataset_path):
        # ID string --> list[Value]
        for filename in os.listdir(tagged_dataset_path):
            filename = os.path.join(tagged_dataset_path, filename)
            print('Reading dataset from', filename, file=sys.stderr)
            with open(filename, 'r', encoding='utf8') as fin:
                header = fin.readline().rstrip('\n').split('\t')
                for line in fin:
                    stuff = dict(zip(header, line.rstrip('\n').split('\t')))
                    # replace the original script "id" with question search "utterance"
                    ex_id = stuff['utterance']
                    original_strings = tsv_unescape_list(stuff['targetValue'])
                    canon_strings = tsv_unescape_list(stuff['targetCanon'])
                    self.target_values_map[ex_id] = to_value_list(
                            original_strings, canon_strings)
        print('Read', len(self.target_values_map), 'examples', file=sys.stderr)

    def evaluate(self, ex_id, predicted_values):
        if ex_id not in self.target_values_map:
            print('WARNING: Example Question "%s" not found' % ex_id)
            return False
        else:
            target_values = self.target_values_map[ex_id]
            predicted_values = to_value_list(predicted_values)
            correct = check_denotation(target_values, predicted_values)
            print(u'%s\t%s\t%s\t%s' % (ex_id, correct,
                    target_values, predicted_values))
            return correct

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tagged-dataset-path',
            default=os.path.join('.', 'tagged', 'data'),
            help='Directory containing CoreNLP-tagged dataset TSV file')
    
    args = parser.parse_args()
    
    evaluator = Evaluator(args.tagged_dataset_path)

    correct = evaluator.evaluate("congressmen re-elected with at least 60% of the vote", ["Wayne Gilchrest","Ben Cardin","Albert Wynn","Steny Hoyer","Roscoe Bartlett","Elijah Cummings"])    
    print(correct)

if __name__ == '__main__':
    main()
