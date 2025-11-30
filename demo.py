"""
Quick Demo Script - Test the AI Data Intelligence Agent Locally
Creates sample data and provides testing commands
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_sample_datasets():
    """Create sample datasets for different domains"""
    
    print("=" * 80)
    print("CREATING SAMPLE DATASETS FOR TESTING")
    print("=" * 80)
    
    # Create data directory
    os.makedirs("sample_data", exist_ok=True)
    
    # 1. Retail Sales Dataset
    print("\n1. Creating retail_sales.csv...")
    dates = pd.date_range(start='2023-01-01', end='2024-01-31', freq='D')
    retail_data = {
        'Date': dates,
        'Sales': np.random.randint(1000, 5000, len(dates)) + np.sin(np.arange(len(dates)) * 0.1) * 500,
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'], len(dates)),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
        'Quantity': np.random.randint(1, 50, len(dates))
    }
    retail_df = pd.DataFrame(retail_data)
    retail_df.to_csv('sample_data/retail_sales.csv', index=False)
    print(f"   ✅ Created: sample_data/retail_sales.csv ({len(retail_df)} rows)")
    
    # 2. Financial Transactions Dataset
    print("\n2. Creating financial_transactions.csv...")
    fin_data = {
        'Date': dates,
        'Amount': np.random.uniform(50, 5000, len(dates)),
        'Category': np.random.choice(['Food', 'Transport', 'Entertainment', 'Shopping', 'Bills'], len(dates)),
        'Account': np.random.choice(['Checking', 'Savings', 'Credit'], len(dates)),
        'Type': np.random.choice(['Debit', 'Credit'], len(dates))
    }
    fin_df = pd.DataFrame(fin_data)
    fin_df.to_csv('sample_data/financial_transactions.csv', index=False)
    print(f"   ✅ Created: sample_data/financial_transactions.csv ({len(fin_df)} rows)")
    
    # 3. Healthcare Patient Data
    print("\n3. Creating healthcare_patients.csv...")
    health_data = {
        'Date': dates,
        'Patients': np.random.randint(20, 100, len(dates)),
        'Department': np.random.choice(['Emergency', 'Surgery', 'Cardiology', 'Pediatrics'], len(dates)),
        'Hospital': np.random.choice(['City General', 'Metro Hospital', 'County Medical'], len(dates)),
        'Admissions': np.random.randint(5, 30, len(dates))
    }
    health_df = pd.DataFrame(health_data)
    health_df.to_csv('sample_data/healthcare_patients.csv', index=False)
    print(f"   ✅ Created: sample_data/healthcare_patients.csv ({len(health_df)} rows)")
    
    # 4. IoT Sensor Data
    print("\n4. Creating iot_sensor_data.csv...")
    iot_data = {
        'Timestamp': dates,
        'Temperature': np.random.uniform(18, 28, len(dates)),
        'Humidity': np.random.uniform(30, 70, len(dates)),
        'Device': np.random.choice(['Sensor_A', 'Sensor_B', 'Sensor_C'], len(dates)),
        'Location': np.random.choice(['Building1', 'Building2', 'Building3'], len(dates))
    }
    iot_df = pd.DataFrame(iot_data)
    iot_df.to_csv('sample_data/iot_sensor_data.csv', index=False)
    print(f"   ✅ Created: sample_data/iot_sensor_data.csv ({len(iot_df)} rows)")
    
    print("\n" + "=" * 80)
    print("✅ ALL SAMPLE DATASETS CREATED!")
    print("=" * 80)
    
    return {
        'retail': 'sample_data/retail_sales.csv',
        'finance': 'sample_data/financial_transactions.csv',
        'healthcare': 'sample_data/healthcare_patients.csv',
        'iot': 'sample_data/iot_sensor_data.csv'
    }

def print_test_instructions(datasets):
    """Print testing instructions"""
    
    print("\n" + "=" * 80)
    print("TESTING INSTRUCTIONS")
    print("=" * 80)
    
    print("\n📋 STEP 1: Setup Environment")
    print("-" * 80)
    print("1. Make sure you have a .env file with your API key:")
    print("   OPENAI_API_KEY=your_key_here")
    print("\n2. Install dependencies (if not done already):")
    print("   pip install -r requirements.txt")
    
    print("\n📋 STEP 2: Run the Application")
    print("-" * 80)
    print("Command:")
    print("   streamlit run app.py")
    print("\nThe app will open at: http://localhost:8501")
    
    print("\n📋 STEP 3: Test with Sample Datasets")
    print("-" * 80)
    
    print("\n🛒 RETAIL SALES TEST:")
    print(f"   File: {datasets['retail']}")
    print("   1. Upload the CSV")
    print("   2. Select 'SALES' as target variable")
    print("   3. Select 'DATE' as date column")
    print("   4. Click 'Run Analysis'")
    print("   5. Ask: 'What are the total sales by region?'")
    
    print("\n💰 FINANCIAL TEST:")
    print(f"   File: {datasets['finance']}")
    print("   1. Upload the CSV")
    print("   2. Select 'AMOUNT' as target variable")
    print("   3. Select 'DATE' as date column")
    print("   4. Ask: 'What are my spending patterns by category?'")
    
    print("\n🏥 HEALTHCARE TEST:")
    print(f"   File: {datasets['healthcare']}")
    print("   1. Upload the CSV")
    print("   2. Select 'PATIENTS' as target variable")
    print("   3. Ask: 'Which department has the highest patient volume?'")
    
    print("\n🌡️ IOT SENSOR TEST:")
    print(f"   File: {datasets['iot']}")
    print("   1. Upload the CSV")
    print("   2. Select 'TEMPERATURE' as target variable")
    print("   3. Ask: 'Are there any temperature anomalies?'")
    
    print("\n📋 STEP 4: Test LLM Agent Capabilities")
    print("-" * 80)
    print("Try these questions with any dataset:")
    print("   • 'What is the total {target}?'")
    print("   • 'Show me the forecast for next week'")
    print("   • 'Which {category} performs best?'")
    print("   • 'What anomalies were detected?'")
    print("   • 'How accurate is the model?'")
    print("   • 'What's the growth trend?'")
    print("   • 'Summarize the key insights'")
    
    print("\n📋 STEP 5: Docker Testing (Optional)")
    print("-" * 80)
    print("Build and run with Docker:")
    print("   docker build -t ai-data-agent .")
    print("   docker run -p 8501:8501 -e OPENAI_API_KEY=your_key ai-data-agent")
    
    print("\n" + "=" * 80)
    print("🎉 READY TO TEST!")
    print("=" * 80)
    print("\nNext: Run 'streamlit run app.py' and upload a sample dataset")
    print("=" * 80)

def verify_environment():
    """Verify environment setup"""
    
    print("\n" + "=" * 80)
    print("ENVIRONMENT CHECK")
    print("=" * 80)
    
    checks = {
        'Python': False,
        'Streamlit': False,
        'Pandas': False,
        'LangChain': False,
        'API Key': False
    }
    
    # Check Python
    try:
        import sys
        print(f"\n✅ Python {sys.version.split()[0]} installed")
        checks['Python'] = True
    except:
        print("\n❌ Python not found")
    
    # Check Streamlit
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} installed")
        checks['Streamlit'] = True
    except:
        print("❌ Streamlit not installed (run: pip install -r requirements.txt)")
    
    # Check Pandas
    try:
        import pandas
        print(f"✅ Pandas {pandas.__version__} installed")
        checks['Pandas'] = True
    except:
        print("❌ Pandas not installed")
    
    # Check LangChain
    try:
        import langchain
        print(f"✅ LangChain {langchain.__version__} installed")
        checks['LangChain'] = True
    except:
        print("❌ LangChain not installed")
    
    # Check API Key
    from dotenv import load_dotenv
    load_dotenv()
    if os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY') or os.getenv('GROQ_API_KEY'):
        print("✅ API Key found in .env")
        checks['API Key'] = True
    else:
        print("⚠️  No API key found in .env (add one to use LLM features)")
    
    print("\n" + "=" * 80)
    all_ready = all(checks.values())
    if all_ready:
        print("🎉 ENVIRONMENT READY!")
    else:
        print("⚠️  Some components missing - see above")
    print("=" * 80)
    
    return all_ready

if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "AI DATA INTELLIGENCE AGENT v6.0" + " " * 32 + "║")
    print("║" + " " * 25 + "DEMO & TEST SCRIPT" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # Check environment
    env_ready = verify_environment()
    
    # Create sample datasets
    datasets = create_sample_datasets()
    
    # Print instructions
    print_test_instructions(datasets)
    
    # Final message
    if env_ready:
        print("\n✨ Everything is ready! Run: streamlit run app.py")
    else:
        print("\n⚠️  Fix environment issues above, then run: streamlit run app.py")
