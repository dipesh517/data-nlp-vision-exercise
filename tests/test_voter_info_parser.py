import pytest
from src.voter_info_parser import VoterInfoParser
from PIL import Image
import pandas as pd
import os
from PIL import Image
from PIL import Image


# Constants for the test cases
VALID_PDF_PATH = "data/eval/electoral_rolls_test.pdf"
INVALID_PDF_PATH = "nonexistent/path/sample.pdf"
SAMPLE_IMAGE_PATH = "data/pdf_images/page_3.png"

# ----------------- Fixtures -------------------


# Utility function to create a sample image (if needed for tests)
def create_sample_image(path):
    image = Image.new('RGB', (100, 100))
    image.save(path)

# Fixture for setting up the VoterInfoParser instance
@pytest.fixture
def voter_info_parser():
    return VoterInfoParser(VALID_PDF_PATH, first_page=3, last_page=4)

# Test cases start here

def test_initialization(voter_info_parser):
    assert voter_info_parser.path == VALID_PDF_PATH
    assert voter_info_parser.first_page == 3
    assert voter_info_parser.last_page == 4

def test_extract_images_from_pdf(voter_info_parser):
    images = voter_info_parser.extract_images_from_pdf()
    assert images  # Should not be empty

def test_extract_voters_info(voter_info_parser):
    voters_info = voter_info_parser.extract_voters_info()
    assert len(voters_info) == 30

def test_save_extracted_voters_info(voter_info_parser):
    voter_info_parser.extract_voters_info()
    csv_file_path = voter_info_parser.save_extracted_voters_info()
    file_path = os.path.basename(csv_file_path)
    assert file_path == os.path.basename(voter_info_parser.path).split('.')[0] + '_output.csv'



if __name__ == "__main__":
    pytest.main()