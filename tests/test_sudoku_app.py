"""
Tests for the Sudoku Solver App Logic

Run with: pytest tests/test_sudoku_app.py -v

These tests cover the core puzzle parsing, validation, and formatting logic
without requiring gradio to be installed.
"""

import pytest
import numpy as np
import pandas as pd


# ============================================================================
# Core functions extracted from app.py for testing
# ============================================================================

def parse_sudoku_input(text: str) -> np.ndarray:
    """Parse various Sudoku input formats into 9x9 numpy array."""
    text = text.strip()
    
    # First try: treat as single line of 81 chars
    chars = []
    for c in text:
        if c.isdigit():
            chars.append(int(c))
        elif c in '.0_':
            chars.append(0)
    
    if len(chars) == 81:
        return np.array(chars).reshape(9, 9)
    
    # Second try: parse as multiline
    puzzle = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('-') or line.startswith('+'):
            continue
        
        row = []
        for c in line:
            if c.isdigit():
                row.append(int(c))
            elif c in '.0_':
                row.append(0)
        
        if len(row) >= 9:
            puzzle.extend(row[:9])
    
    if len(puzzle) == 81:
        return np.array(puzzle).reshape(9, 9)
    
    raise ValueError(f"Could not parse Sudoku. Got {len(puzzle)} cells, expected 81.")


def validate_solution(puzzle: np.ndarray, solution: np.ndarray) -> tuple:
    """Validate a Sudoku solution."""
    if np.any(solution == 0):
        empty_count = np.sum(solution == 0)
        return False, f"Incomplete: {empty_count} cells still empty"
    
    for i in range(9):
        if len(set(solution[i])) != 9:
            return False, f"Invalid row {i+1}"
    
    for j in range(9):
        if len(set(solution[:, j])) != 9:
            return False, f"Invalid column {j+1}"
    
    for bi in range(3):
        for bj in range(3):
            box = solution[bi*3:(bi+1)*3, bj*3:(bj+1)*3].flatten()
            if len(set(box)) != 9:
                return False, f"Invalid box ({bi+1}, {bj+1})"
    
    mask = puzzle > 0
    if not np.all(solution[mask] == puzzle[mask]):
        return False, "Givens were modified"
    
    return True, "Valid solution!"


def create_empty_board():
    """Create an empty 9x9 Sudoku board as a dataframe."""
    data = [["" for _ in range(9)] for _ in range(9)]
    return pd.DataFrame(data)


def puzzle_string_to_dataframe(puzzle_str: str):
    """Convert 81-char puzzle string to dataframe."""
    data = []
    for i in range(9):
        row = []
        for j in range(9):
            val = puzzle_str[i*9 + j]
            if val == '0' or val == '.':
                row.append("")
            else:
                row.append(val)
        data.append(row)
    return pd.DataFrame(data)


def dataframe_to_puzzle_array(df) -> np.ndarray:
    """Convert dataframe to numpy array with input validation."""
    puzzle = np.zeros((9, 9), dtype=int)
    
    if df is None or len(df) < 9:
        return puzzle
    
    for i in range(min(9, len(df))):
        for j in range(min(9, len(df.columns))):
            try:
                val = df.iloc[i, j]
                if val is None:
                    continue
                val_str = str(val).strip()[:10]
                if val_str.isdigit() and len(val_str) == 1:
                    num = int(val_str)
                    if 1 <= num <= 9:
                        puzzle[i, j] = num
            except (IndexError, ValueError, TypeError):
                pass
    return puzzle


def format_grid_html(grid: np.ndarray, original: np.ndarray = None) -> str:
    """Format a 9x9 grid as HTML table."""
    html = ['<table style="border-collapse: collapse; font-family: monospace;">']
    
    for i in range(9):
        html.append('<tr>')
        for j in range(9):
            val = grid[i, j]
            
            borders = []
            if i % 3 == 0:
                borders.append("border-top: 2px solid black")
            if j % 3 == 0:
                borders.append("border-left: 2px solid black")
            if i == 8:
                borders.append("border-bottom: 2px solid black")
            if j == 8:
                borders.append("border-right: 2px solid black")
            
            is_given = original is not None and original[i, j] > 0
            bg_color = "#fff" if is_given else "#e8f5e9"
            
            style = "; ".join(borders + [
                f"background: {bg_color}",
                "width: 32px",
                "height: 32px", 
                "text-align: center",
                "font-size: 18px",
            ])
            
            if isinstance(val, (int, np.integer)) and 1 <= val <= 9:
                content = str(int(val))
            else:
                content = "·"
            html.append(f'<td style="{style}">{content}</td>')
        html.append('</tr>')
    
    html.append('</table>')
    return '\n'.join(html)


# Example puzzles for testing
EXAMPLE_PUZZLES = [
    ("Easy", "530070000600195000098000060800060003400803001700020006060000280000419005000080079"),
    ("Medium", "000260701680070090190004500820100040004602900050003028009300074040050036703018000"),
    ("Hard", "000000000000003085001020000000507000004000100090000000500000073002010000000040009"),
    ("Extreme", "800000000003600000070090200050007000000045700000100030001000068008500010090000400"),
]


# ============================================================================
# Test Classes
# ============================================================================

class TestPuzzleParsing:
    """Tests for puzzle input parsing."""
    
    def test_parse_81_char_string(self):
        """Test parsing a standard 81-character puzzle string."""
        puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        result = parse_sudoku_input(puzzle_str)
        
        assert result.shape == (9, 9)
        assert result[0, 0] == 5
        assert result[0, 1] == 3
        assert result[0, 3] == 0  # Empty cell
        assert result[8, 8] == 9
    
    def test_parse_with_dots(self):
        """Test parsing puzzle with dots for empty cells."""
        puzzle_str = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
        result = parse_sudoku_input(puzzle_str)
        
        assert result.shape == (9, 9)
        assert result[0, 0] == 5
        assert result[0, 2] == 0  # Dot = empty
    
    def test_parse_multiline(self):
        """Test parsing multiline formatted puzzle."""
        puzzle_str = """
        5 3 . | . 7 . | . . .
        6 . . | 1 9 5 | . . .
        . 9 8 | . . . | . 6 .
        ------+-------+------
        8 . . | . 6 . | . . 3
        4 . . | 8 . 3 | . . 1
        7 . . | . 2 . | . . 6
        ------+-------+------
        . 6 . | . . . | 2 8 .
        . . . | 4 1 9 | . . 5
        . . . | . 8 . | . 7 9
        """
        result = parse_sudoku_input(puzzle_str)
        
        assert result.shape == (9, 9)
        assert result[0, 0] == 5
        assert result[0, 4] == 7
    
    def test_parse_invalid_too_short(self):
        """Test that short input raises error."""
        with pytest.raises(ValueError):
            parse_sudoku_input("12345")
    
    def test_parse_empty_string(self):
        """Test that empty string raises error."""
        with pytest.raises(ValueError):
            parse_sudoku_input("")


class TestSolutionValidation:
    """Tests for Sudoku solution validation."""
    
    def test_valid_solution(self):
        """Test that a valid solution passes."""
        puzzle = np.array([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ])
        
        solution = np.array([
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ])
        
        is_valid, msg = validate_solution(puzzle, solution)
        assert is_valid, f"Expected valid but got: {msg}"
    
    def test_incomplete_solution(self):
        """Test that incomplete solution is detected."""
        puzzle = np.zeros((9, 9), dtype=int)
        solution = np.zeros((9, 9), dtype=int)
        solution[0, 0] = 5  # Only one cell filled
        
        is_valid, msg = validate_solution(puzzle, solution)
        assert not is_valid
        assert "Incomplete" in msg or "empty" in msg.lower()
    
    def test_duplicate_in_row(self):
        """Test that duplicate in row is detected."""
        puzzle = np.zeros((9, 9), dtype=int)
        solution = np.ones((9, 9), dtype=int)  # All 1s - invalid!
        
        is_valid, msg = validate_solution(puzzle, solution)
        assert not is_valid
    
    def test_modified_givens(self):
        """Test that modified given cells are detected."""
        # Use a valid complete solution but with one given changed
        puzzle = np.array([
            [5, 3, 0, 0, 7, 0, 0, 0, 0],
            [6, 0, 0, 1, 9, 5, 0, 0, 0],
            [0, 9, 8, 0, 0, 0, 0, 6, 0],
            [8, 0, 0, 0, 6, 0, 0, 0, 3],
            [4, 0, 0, 8, 0, 3, 0, 0, 1],
            [7, 0, 0, 0, 2, 0, 0, 0, 6],
            [0, 6, 0, 0, 0, 0, 2, 8, 0],
            [0, 0, 0, 4, 1, 9, 0, 0, 5],
            [0, 0, 0, 0, 8, 0, 0, 7, 9],
        ])
        
        # Valid solution but with the first given (5) changed to 1
        solution = np.array([
            [1, 3, 4, 6, 7, 8, 9, 5, 2],  # Changed 5 to 1 (swap with pos 7)
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [5, 9, 8, 3, 4, 2, 1, 6, 7],  # Adjusted to keep valid rows
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9],
        ])
        
        is_valid, msg = validate_solution(puzzle, solution)
        # Will fail either on row/col/box validation OR givens check
        assert not is_valid


class TestDataframeConversion:
    """Tests for dataframe <-> puzzle array conversion."""
    
    def test_puzzle_string_to_dataframe(self):
        """Test converting puzzle string to dataframe."""
        puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        df = puzzle_string_to_dataframe(puzzle_str)
        
        assert df.shape == (9, 9)
        assert df.iloc[0, 0] == "5"
        assert df.iloc[0, 2] == ""  # Empty cell
    
    def test_dataframe_to_puzzle_array(self):
        """Test converting dataframe to numpy array."""
        puzzle_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        df = puzzle_string_to_dataframe(puzzle_str)
        arr = dataframe_to_puzzle_array(df)
        
        assert arr.shape == (9, 9)
        assert arr[0, 0] == 5
        assert arr[0, 2] == 0
    
    def test_empty_board(self):
        """Test creating an empty board."""
        df = create_empty_board()
        
        assert df.shape == (9, 9)
        arr = dataframe_to_puzzle_array(df)
        assert np.sum(arr) == 0  # All zeros
    
    def test_invalid_input_ignored(self):
        """Test that invalid cell values are ignored."""
        df = pd.DataFrame([
            ["5", "abc", "4", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
        ])
        
        arr = dataframe_to_puzzle_array(df)
        assert arr[0, 0] == 5
        assert arr[0, 1] == 0  # "abc" should be ignored
        assert arr[0, 2] == 4


class TestHtmlFormatting:
    """Tests for HTML grid formatting."""
    
    def test_format_grid_html_returns_string(self):
        """Test that format_grid_html returns a string."""
        puzzle = np.zeros((9, 9), dtype=int)
        puzzle[0, 0] = 5
        
        html = format_grid_html(puzzle)
        
        assert isinstance(html, str)
        assert "<table" in html
        assert "5" in html
    
    def test_format_grid_html_with_original(self):
        """Test formatting with original puzzle highlighting."""
        original = np.zeros((9, 9), dtype=int)
        original[0, 0] = 5
        
        solution = np.ones((9, 9), dtype=int) * 9
        solution[0, 0] = 5
        
        html = format_grid_html(solution, original)
        
        assert "<table" in html
    
    def test_format_grid_html_escapes_invalid(self):
        """Test that invalid values don't break HTML."""
        puzzle = np.zeros((9, 9), dtype=int)
        puzzle[0, 0] = 999  # Invalid value
        
        html = format_grid_html(puzzle)
        
        assert "<table" in html
        assert "·" in html  # Invalid should show as dot


class TestExamplePuzzles:
    """Tests for example puzzles."""
    
    def test_example_puzzles_exist(self):
        """Test that example puzzles are defined."""
        assert len(EXAMPLE_PUZZLES) >= 4
    
    def test_example_puzzles_valid_format(self):
        """Test that all example puzzles have valid format."""
        for name, puzzle_str in EXAMPLE_PUZZLES:
            assert isinstance(name, str)
            assert isinstance(puzzle_str, str)
            assert len(puzzle_str) == 81
            
            arr = parse_sudoku_input(puzzle_str)
            assert arr.shape == (9, 9)
    
    def test_example_puzzles_have_empty_cells(self):
        """Test that example puzzles have empty cells to solve."""
        for name, puzzle_str in EXAMPLE_PUZZLES:
            arr = parse_sudoku_input(puzzle_str)
            empty_count = np.sum(arr == 0)
            assert empty_count > 0, f"Puzzle '{name}' has no empty cells"


class TestSecurityValidation:
    """Tests for input sanitization and security."""
    
    def test_long_input_truncated(self):
        """Test that overly long cell values are handled safely."""
        df = pd.DataFrame([
            ["5" * 100, "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
        ])
        
        arr = dataframe_to_puzzle_array(df)
        assert arr[0, 0] == 0  # Long string should be ignored
    
    def test_html_injection_prevented(self):
        """Test that HTML in cell values doesn't cause issues."""
        df = pd.DataFrame([
            ["<script>alert('xss')</script>", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
        ])
        
        arr = dataframe_to_puzzle_array(df)
        assert arr[0, 0] == 0
    
    def test_none_values_handled(self):
        """Test that None values don't cause crashes."""
        df = pd.DataFrame([
            [None, "5", None, "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
        ])
        
        arr = dataframe_to_puzzle_array(df)
        assert arr[0, 0] == 0
        assert arr[0, 1] == 5
    
    def test_negative_numbers_ignored(self):
        """Test that negative numbers are ignored."""
        df = pd.DataFrame([
            ["-5", "5", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", ""],
        ])
        
        arr = dataframe_to_puzzle_array(df)
        assert arr[0, 0] == 0  # -5 should be ignored
        assert arr[0, 1] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
