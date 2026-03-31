"""Test Enhanced Analyst Chain with Multiple Datasets

This script tests the enhanced analyst chain with:
1. Sample sales data
2. Generated test datasets
3. Various query types
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from ai_data_analyst.chains import EnhancedAnalystChain, QueryClassifier
from ai_data_analyst.config import GOOGLE_API_KEY


def create_test_datasets():
    """Create various test datasets for comprehensive testing."""
    
    # Dataset 1: Sales data (already exists, but let's create variations)
    sales_data = pd.DataFrame({
        'product': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Monitor'] * 10,
        'category': ['Electronics'] * 25 + ['Furniture'] * 25,
        'region': ['North', 'South', 'East', 'West'] * 12 + ['North', 'South'],
        'quarter': ['Q1', 'Q2', 'Q3', 'Q4'] * 12 + ['Q1', 'Q2'],
        'sales': np.random.uniform(1000, 50000, 50),
        'quantity': np.random.randint(10, 200, 50),
        'profit': np.random.uniform(500, 15000, 50),
        'customer_rating': np.random.uniform(3.5, 5.0, 50),
        'discount': np.random.uniform(0.01, 0.30, 50),
    })
    
    # Dataset 2: Customer data
    np.random.seed(42)
    n_customers = 100
    customer_data = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'age': np.random.randint(18, 70, n_customers),
        'gender': np.random.choice(['M', 'F', 'Other'], n_customers),
        'income': np.random.normal(60000, 20000, n_customers).clip(20000, 150000),
        'spending_score': np.random.randint(1, 100, n_customers),
        'visits_per_month': np.random.poisson(5, n_customers),
        'total_purchases': np.random.exponential(500, n_customers),
        'satisfaction': np.random.choice(['Low', 'Medium', 'High'], n_customers),
        'churn': np.random.choice([0, 1], n_customers, p=[0.7, 0.3]),
    })
    
    # Dataset 3: Time series data
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    time_series_data = pd.DataFrame({
        'date': dates,
        'daily_sales': np.random.normal(5000, 1000, 365).clip(1000, 10000),
        'website_visitors': np.random.poisson(1000, 365),
        'orders': np.random.poisson(50, 365),
        'conversion_rate': np.random.uniform(0.02, 0.08, 365),
        'marketing_spend': np.random.uniform(100, 1000, 365),
    })
    
    # Add trend and seasonality
    time_series_data['daily_sales'] += np.sin(np.arange(365) * 2 * np.pi / 365) * 500
    time_series_data['daily_sales'] += np.arange(365) * 2  # Upward trend
    
    return {
        'sales': sales_data,
        'customer': customer_data,
        'time_series': time_series_data,
    }


def test_query_classifier():
    """Test the query classifier with various queries."""
    print("\n" + "="*60)
    print("Testing Query Classifier")
    print("="*60)
    
    test_queries = [
        "What is the total sales?",
        "Which region has the highest profit?",
        "Show me the trend over time",
        "What's the distribution of customer ratings?",
        "Is there a correlation between sales and profit?",
        "What are the top 5 products?",
        "Filter customers with high income",
        "Compare North vs South regions",
    ]
    
    results = []
    for query in test_queries:
        query_type, confidence = QueryClassifier.classify(query)
        results.append((query, query_type, confidence))
        print(f"\nQuery: {query}")
        print(f"  -> Type: {query_type} (confidence: {confidence:.2f})")
    
    return results


def test_enhanced_chain(df: pd.DataFrame, dataset_name: str, test_queries: list):
    """Test enhanced analyst chain with a dataset and queries."""
    print("\n" + "="*60)
    print(f"Testing Enhanced Analyst Chain - {dataset_name}")
    print("="*60)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize enhanced chain
    try:
        chain = EnhancedAnalystChain(df, GOOGLE_API_KEY)
    except Exception as e:
        print(f"❌ Failed to initialize chain: {e}")
        return []
    
    results = []
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-'*60}")
        print(f"Query {i}: {query}")
        print(f"{'-'*60}")
        
        try:
            # Run analysis
            response = chain.analyze(query)
            
            # Print results
            print(f"\n✅ Query Type: {response.query_type}")
            print(f"📈 Confidence: {response.confidence_score:.1%}")
            print(f"\n💬 Answer: {response.final_answer[:300]}...")
            
            if response.chart_type:
                print(f"\n📊 Suggested Chart: {response.chart_type}")
                if response.chart_columns:
                    print(f"   Columns: {', '.join(response.chart_columns)}")
            
            # Get workflow summary
            print(f"\n🔍 Workflow Summary:")
            summary = chain.get_workflow_summary(response)
            print(summary[:500])
            
            # Get follow-up suggestions
            follow_ups = chain.suggest_follow_up_questions(response)
            if follow_ups:
                print(f"\n💡 Follow-up Questions:")
                for fu in follow_ups:
                    print(f"  • {fu}")
            
            results.append({
                'query': query,
                'response': response,
                'success': True,
            })
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'query': query,
                'error': str(e),
                'success': False,
            })
    
    return results


def run_comprehensive_tests():
    """Run comprehensive tests on all datasets."""
    print("\n" + "🚀"*30)
    print("Starting Comprehensive Enhanced Analyst Tests")
    print("🚀"*30)
    
    # Create test datasets
    datasets = create_test_datasets()
    
    # Define test queries for each dataset
    test_queries_by_dataset = {
        'sales': [
            "What is the total sales revenue?",
            "Which product has the highest average profit?",
            "Compare sales between North and South regions",
            "What's the correlation between discount and sales?",
            "Show me the top 3 products by customer rating",
        ],
        'customer': [
            "What is the average income of customers?",
            "Is there a correlation between age and spending score?",
            "What's the distribution of satisfaction levels?",
            "Which gender has higher average purchases?",
            "What factors are associated with customer churn?",
        ],
        'time_series': [
            "What is the trend in daily sales over time?",
            "Calculate the average conversion rate",
            "Is there a correlation between marketing spend and orders?",
            "Show me the distribution of website visitors",
            "What's the total sales for the year?",
        ],
    }
    
    all_results = {}
    
    # Test each dataset
    for dataset_name, df in datasets.items():
        queries = test_queries_by_dataset[dataset_name]
        results = test_enhanced_chain(df, dataset_name, queries)
        all_results[dataset_name] = results
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for dataset_name, results in all_results.items():
        success_count = sum(1 for r in results if r.get('success', False))
        total_count = len(results)
        print(f"\n{dataset_name}: {success_count}/{total_count} queries successful")
        
        if success_count < total_count:
            print("  Failed queries:")
            for r in results:
                if not r.get('success', False):
                    print(f"    • {r.get('query', 'Unknown')} - {r.get('error', 'Unknown error')}")
    
    return all_results


def test_with_sample_data():
    """Test with the actual sample data file."""
    print("\n" + "="*60)
    print("Testing with Sample Data File")
    print("="*60)
    
    # Load sample data
    sample_path = Path(__file__).parent.parent / 'data' / 'sample_data.csv'
    
    if not sample_path.exists():
        print(f"❌ Sample data not found at {sample_path}")
        return None
    
    df = pd.read_csv(sample_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Test queries
    test_queries = [
        "What is the total sales across all products?",
        "Which category has the highest average profit?",
        "Compare sales between Q1 and Q2",
        "What's the correlation between sales and quantity?",
        "Show me the top 5 products by revenue",
        "What is the average customer rating by category?",
    ]
    
    results = test_enhanced_chain(df, 'sample_data', test_queries)
    return results


if __name__ == "__main__":
    # Test query classifier
    classifier_results = test_query_classifier()
    
    # Test with generated datasets
    comprehensive_results = run_comprehensive_tests()
    
    # Test with sample data
    sample_results = test_with_sample_data()
    
    print("\n" + "✅"*30)
    print("All tests completed!")
    print("✅"*30)
