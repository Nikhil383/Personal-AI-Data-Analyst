import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from data_analysis import analyst

class TestWorkflow(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'B'],
            'Value': [10, 20, 30, 40],
            'Date': pd.date_range('2024-01-01', periods=4)
        })

    def test_deterministic_prompt(self):
        """Test that a known prompt returns immediate code without LLM."""
        prompt = "Summarize the dataset"
        code = analyst.prompt_to_code(prompt, self.df)
        self.assertIsNotNone(code)
        self.assertIn("info = []", code)

    @patch('data_analysis.analyst.ask_llm')
    def test_llm_generation_structure(self, mock_ask_llm):
        """Test that generate_analysis_code calls the LLM with correct context and parses result."""
        # Mock LLM response
        mock_ask_llm.return_value = "```python\nresult = df['Value'].mean()\n```"
        
        code = analyst.generate_analysis_code(self.df, "Calculate mean")
        
        # Verify LLM was called
        mock_ask_llm.assert_called_once()
        args, _ = mock_ask_llm.call_args
        prompt_sent = args[0]
        
        # Check that context was included
        self.assertIn("DataFrame Columns:", prompt_sent)
        self.assertIn("- Category (object)", prompt_sent)
        
        # Check parsing
        self.assertEqual(code, "result = df['Value'].mean()")

    def test_run_code_execution(self):
        """Test that valid python code executes and returns result."""
        code = "result = df['Value'].sum()"
        res = analyst.run_code(self.df, code)
        self.assertEqual(res['type'], 'text')
        self.assertEqual(float(res['output']), 100.0)

if __name__ == '__main__':
    unittest.main()
