import subprocess
import sys
from pathlib import Path

def main():
    try:
        # Check if we're in a git repository
        subprocess.run(["git", "rev-parse", "--git-dir"], check=True, capture_output=True)
        
        # Install pre-commit hooks
        result = subprocess.run(["pre-commit", "install"], check=True)
        if result.returncode == 0:
            print("Pre-commit hooks installed successfully!")
    except subprocess.CalledProcessError:
        print("Warning: Not a git repository or pre-commit installation failed")
        sys.exit(0)

if __name__ == "__main__":
    main() 