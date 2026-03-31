"""Demo script to showcase enhanced analyst features with sample data"""
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ai_data_analyst.chains import QueryClassifier
from ai_data_analyst.config import GOOGLE_API_KEY

print("="*60)
print("Enhanced AI Data Analyst - Feature Demo")
print("="*60)

# 1. Demo Query Classification
print("\n1. QUERY CLASSIFICATION")
print("-"*60)

test_queries = [
    "What is the total sales?",
    "Which region has the highest profit?",
    "Show me the trend over time",
    "What's the distribution of ratings?",
    "Is there a correlation between sales and profit?",
    "What are the top 5 products?",
    "Filter customers with high income",
    "Compare North vs South regions",
]

for query in test_queries:
    query_type, confidence = QueryClassifier.classify(query)
    confidence_badge = "✅" if confidence >= 0.67 else "⚠️" if confidence >= 0.33 else "❌"
    print(f"{confidence_badge} {query_type:15s} | {query}")

# 2. Load sample data and test with real API (limited)
print("\n2. TESTING WITH SAMPLE DATA")
print("-"*60)

sample_path = Path(__file__).parent.parent / 'data' / 'sample_data.csv'
# Read the CSV, skipping the docstring line at the top
df = pd.read_csv(sample_path, skiprows=1)

print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {', '.join(df.columns)}")

# Show data summary
print("\nData Summary:")
print(f"  - Total records: {len(df):,}")
print(f"  - Numeric columns: {len(df.select_dtypes(include=['number']))}")
if 'category' in df.columns:
    print(f"  - Categories: {df['category'].nunique()}")
if 'region' in df.columns:
    print(f"  - Regions: {df['region'].nunique()}")
if 'product' in df.columns:
    print(f"  - Products: {df['product'].nunique()}")

# 3. Show what the enhanced chain provides
print("\n3. ENHANCED FEATURES OVERVIEW")
print("-"*60)

print("""
The EnhancedAnalystChain provides:

✅ CONFIDENCE SCORING (0-100%)
   - High (≥80%): Reliable answer
   - Medium (50-79%): Some uncertainty  
   - Low (<50%): Review carefully

✅ QUERY CLASSIFICATION
   - Aggregation: total, sum, average, count
   - Comparison: compare, vs, higher, lower
   - Trend: trend, over time, change
   - Distribution: distribution, spread
   - Correlation: correlation, relationship
   - Top N: top, best, highest
   - Filter: filter, where, which

✅ VALIDATION
   - Checks for NaN/None values
   - Validates number ranges
   - Detects error messages

✅ FOLLOW-UP SUGGESTIONS
   - Context-aware questions
   - Based on query type
   - Encourages deeper analysis

✅ DETAILED REASONING
   - Step-by-step breakdown
   - Confidence per step
   - Transparent workflow
""")

# 4. Usage example
print("\n4. USAGE EXAMPLE")
print("-"*60)

print("""
# Initialize enhanced chain
from ai_data_analyst.chains import EnhancedAnalystChain

chain = EnhancedAnalystChain(df, GOOGLE_API_KEY)

# Analyze with confidence scoring
response = chain.analyze("What is the total sales?")

print(f"Answer: {response.final_answer}")
print(f"Confidence: {response.confidence_score:.0%}")
print(f"Query Type: {response.query_type}")

# Get follow-up suggestions
follow_ups = chain.suggest_follow_up_questions(response)
for fu in follow_ups:
    print(f"  - {fu}")

# View reasoning steps
for step in response.reasoning_steps:
    print(f"Step {step.step}: {step.action} - {step.thought}")
""")

# 5. Run the enhanced app
print("\n5. TO RUN THE ENHANCED APP")
print("-"*60)
print("""
.venv\\Scripts\\activate
streamlit run src/ai_data_analyst/main_enhanced.py

The enhanced app provides:
- Toggle between Standard and Enhanced modes
- Confidence badges on all answers
- Reasoning step visualization
- Follow-up question buttons
- Validation feedback
""")

print("\n" + "="*60)
print("Demo Complete!")
print("="*60)
