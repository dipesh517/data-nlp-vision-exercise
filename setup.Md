# Clone the repo 

Install necessary libraries in the system 

If you're using MACOs
    brew install poppler
    brew install tesseract
    brew install opencv

If you're using Linux
    sudo apt-get install poppler-utils tesseract-ocr


# Create a conda environment inside the working directory
 conda create --prefix ./env  python=3.8
# Activate the conda environment

# Install the requirements.txt inside the environment
pip install -r requirements.txt


# To parse the voter info 
python3 src/voter_info_parser.py

