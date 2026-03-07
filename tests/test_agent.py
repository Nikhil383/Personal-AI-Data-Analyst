import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).parent.parent / "src" / "ai_data_analyst"))

import pandas as pd
from ai_data_analyst.chains.analyst_chain import AnalystChain

def test_agent():
    # Load env vars manually
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in environment.")
        return

    # Create dummy data
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "Age": [25, 30, 35, 40, 45],
        "Salary": [50000, 60000, 70000, 80000, 90000],
        "Department": ["HR", "IT", "HR", "IT", "Sales"]
    }
    df = pd.DataFrame(data)

    print("Initializing AnalystChain...")
    chain = AnalystChain(df, api_key)
    
    print("\nTesting Query 1: What is the average age?")
    res1 = chain.query("What is the average age?")
    print("Response 1:", res1)
    
    print("\nTesting Query 2: What is the total salary for the IT department?")
    res2 = chain.query("What is the total salary for the IT department?")
    print("Response 2:", res2)

    print("\nAgent testing complete!")

if __name__ == "__main__":
    test_agent()
