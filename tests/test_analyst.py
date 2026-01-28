import pandas as pd
import numpy as np
import sys
import os

# Add src to path if running directly
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data_analysis.analyst import suggest_prompts, prompt_to_code

def test_suggest_prompts():
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.choice(['cat', 'dog'], size=100),
        'date': pd.date_range('2025-01-01', periods=100)
    })
    suggestions = suggest_prompts(df)
    assert len(suggestions) > 0
    # Check for specific expected suggestions
    assert any("Summarize the dataset" in s for s in suggestions)
    assert any("histogram" in s for s in suggestions) # Since 'A' is numeric

def test_prompt_to_code_deterministic():
    df = pd.DataFrame({
        'val': [1, 2, 3, 4, 5]
    })
    prompt = "Create a histogram of the numeric column 'val'."
    code = prompt_to_code(prompt, df)
    assert code is not None
    assert "plt.figure" in code
    assert "df['val']" in code

def test_prompt_to_code_unknown():
    df = pd.DataFrame({'a': [1]})
    prompt = "Do something random that you don't know how to do"
    code = prompt_to_code(prompt, df)
    assert code is None
