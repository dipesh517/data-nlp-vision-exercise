from pytesseract import image_to_string
from pytesseract import Output
from PIL import ImageEnhance, ImageFilter

import fitz  # PyMuPDF
import io
from PIL import Image
import re

class VoterInfoParser:
    def __init__(self, path, first_page=0, last_page=None):
        self.path = path
        self.first_page = first_page
        self.last_page = last_page
        self.refined_parsed_voters = []

    # Correcting the method to unpack and extract images from the PDF

    def extract_images_from_pdf(self):
        pdf_path = self.path
        first_page = self.first_page
        last_page = self.last_page
        images = []
        # Open the PDF file with PyMuPDF
        pdf_file = fitz.open(pdf_path)
        
        # Go through each page
        for page_number in range(first_page, last_page):
            page = pdf_file[page_number]
            
            # List of images on the page (the image_list contains more info than we assumed)
            image_list = page.get_images(full=True)
            
            for img in image_list:  # img contains several pieces of info, including xref
                image_index = img[0]  # xref is the first item, it's the unique identifier of the image
                base_image = pdf_file.extract_image(image_index)
                image_bytes = base_image["image"]
                
                # Convert to a PIL image format from bytes for further processing
                pil_image = Image.open(io.BytesIO(image_bytes))
                images.append(pil_image)
        
        return images
    

    def preprocess_image_adjusted(self,image):
        """
        Apply preprocessing steps to an image to make it more suitable for OCR, with adjustments to ensure compatibility.
        """
        # Resize the image to increase the size of the text
        image = image.resize((image.width * 3, image.height * 3 ), Image.LANCZOS)

        # Convert the image to grayscale
        image = image.convert('L') 

        # Enhance contrast. This is done on the grayscale image, not the binary image.
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Increase contrast

        # Apply binarization to convert the image to black and white
        image = image.point(lambda x: 0 if x < 128 else 255, '1') 

        # Apply sharpening to make text boundaries more distinct
        image = image.filter(ImageFilter.SHARPEN)

        return image


    def ocr_pdf(self):
        # Apply the adjusted preprocessing to the extracted images
        preprocessed_images_adjusted = [self.preprocess_image_adjusted(img) for img in self.extract_images_from_pdf()]

        # Re-apply OCR to the preprocessed images and collect the results
        preprocessed_ocr_results_adjusted = []
        for img in preprocessed_images_adjusted:
            result = image_to_string(img, output_type=Output.STRING)
            preprocessed_ocr_results_adjusted.append(result)

        # Return the OCR results after adjusted preprocessing
        return preprocessed_ocr_results_adjusted
    

    def parse_voter_info(self,raw_text):
        """
        This function takes the raw text data from the OCR and parses out the voter information.
        It uses regular expressions to identify patterns in the text corresponding to the information we want to extract.
        """
        voters = []

        # Split the text by voter entries. We're assuming that each voter entry starts with a digit and a space.
        # This might need to be adjusted depending on the actual text structure.
        voter_entries = re.split(r'(?<=\n)(?=\d+\s)', raw_text)

        for entry in voter_entries:
            voter = {}
            
            # Search for the voter's name. We assume that the line with 'Name:' contains the name.
            name_match = re.search(r'Name:\s*(.*?)(?=\n)', entry)
            if name_match:
                voter['Name'] = name_match.group(1).strip()

            # Search for the father's or husband's name.
            # We look for either "Father's Name:" or "Husband's Name:" or "Mother's Name:"
            relative_match = re.search(r"(Father's Name|Husband's Name|Mother's Name):\s*(.*?)(?=\n)", entry)
            if relative_match:
                relation_type = relative_match.group(1).strip()
                relative_name = relative_match.group(2).strip()
                voter[relation_type] = relative_name

            # Search for the house number. We assume it's on the line with 'House Number:'.
            house_no_match = re.search(r'House Number\s*:\s*(.*?)(?=\n)', entry)
            if house_no_match:
                voter['House Number'] = house_no_match.group(1).strip()

            # Search for age. We assume it's on the line with 'Age:'.
            age_match = re.search(r'Age:\s*(\d+)', entry)
            if age_match:
                voter['Age'] = int(age_match.group(1).strip())

            # Search for gender. We assume it's on the line with 'Gender:'.
            gender_match = re.search(r'Gender:\s*(MALE|FEMALE)', entry)
            if gender_match:
                voter['Gender'] = gender_match.group(1).strip()

            # If we found a name, we assume this is a valid entry.
            if 'Name' in voter:
                voters.append(voter)

        return voters
    
    def extract_voters_info(self):
        """
        This function combines the OCR and parsing steps to extract the voter information from the PDF.
        """
        preprocessed_ocr_results_adjusted = self.ocr_pdf()
        for text in preprocessed_ocr_results_adjusted:
            parsed_voters = self.parse_voter_info(text)
            self.refined_parsed_voters.extend(parsed_voters)
        return self.refined_parsed_voters


if __name__ == '__main__':
    import os 
    from common.path_setup import data_dir

    # Path to your PDF file
    pdf_file_path = os.path.join(data_dir, 'electoral_rolls.pdf')

    voter_info_parser = VoterInfoParser(pdf_file_path, first_page=3, last_page=4)

    extracted_text = voter_info_parser.extract_voters_info()

    print(f"""extracted_text>>> {extracted_text}""")




