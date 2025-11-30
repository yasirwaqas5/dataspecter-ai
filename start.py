"""
Quick Start Script - Sets up and runs the AI Data Intelligence Agent
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    print("="*80)
    print("AI DATA INTELLIGENCE AGENT v6.0 - QUICK START")
    print("="*80)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher required")
        print(f"   Current version: {sys.version}")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected\n")
    
    # Check if venv exists
    venv_path = Path("venv")
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"])
        print("✅ Virtual environment created\n")
    
    # Determine pip path
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Install requirements
    print("📥 Installing dependencies...")
    print("   This may take a few minutes...")
    result = subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("❌ Installation failed:")
        print(result.stderr)
        return
    
    print("✅ Dependencies installed\n")
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  No .env file found")
        print("   Creating from template...")
        template = Path(".env.example")
        if template.exists():
            import shutil
            shutil.copy(template, env_file)
            print("✅ Created .env file")
            print("\n" + "="*80)
            print("⚠️  IMPORTANT: Add your API key to the .env file before running!")
            print("="*80)
            print("\nEdit .env and add ONE of these API keys:")
            print("  - OPENAI_API_KEY (from https://platform.openai.com/api-keys)")
            print("  - ANTHROPIC_API_KEY (from https://console.anthropic.com/)")
            print("  - GROQ_API_KEY (from https://console.groq.com/)")
            print("\nThen run: streamlit run app.py")
            return
    
    print("✅ Configuration found\n")
    
    # Run Streamlit
    print("="*80)
    print("🚀 STARTING APPLICATION...")
    print("="*80)
    print("\n📱 The app will open in your browser at http://localhost:8501")
    print("   Press Ctrl+C to stop\n")
    
    subprocess.run([str(python_path), "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
