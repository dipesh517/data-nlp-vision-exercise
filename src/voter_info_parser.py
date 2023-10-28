from pytesseract import image_to_string
from pytesseract import Output
from PIL import ImageEnhance, ImageFilter
import numpy as np
import fitz  # PyMuPDF
import io
from PIL import Image
import re
import os 
import cv2
import pandas as pd
import random

from src.common.path_setup import *
from src.common.logger import logger as sc_logger
from src.common.path_setup import data_dir

class VoterInfoParser:
    def __init__(self, path, first_page=0, last_page=None):
        self.path = path
        self.first_page = first_page
        self.last_page = last_page
        self.refined_parsed_voters = []
        self.validated_voters_df = None

    # Correcting the method to unpack and extract images from the PDF

    def extract_images_from_pdf(self):
        pdf_path = self.path
        first_page = self.first_page
        last_page = self.last_page
        images = []
        # Open the PDF file with PyMuPDF
        pdf_file = fitz.open(pdf_path)

        sc_logger.info(f"extraction of images from pdf started>>>>")
        
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
        
        sc_logger.info(f"No of pages in pdf>>> {len(images)}")

        sc_logger.info(f"extraction of images from pdf completed>>>>")
        return images
    

    def preprocess_image_adjusted(self,image):
        """
        Apply preprocessing steps to an image to make it more suitable for OCR, with adjustments to ensure compatibility.
        """
        # Resize the image to increase the size of the text
        # image = image.resize((image.width * 5, image.height * 5), Image.LANCZOS)

        # # Convert the image to grayscale
        image = image.convert('L') 

        # # Enhance contrast. This is done on the grayscale image, not the binary image.
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Increase contrast

        # # Apply binarization to convert the image to black and white
        image = image.point(lambda x: 0 if x < 128 else 255, '1') 

        # # Apply sharpening to make text boundaries more distinct
        image = image.filter(ImageFilter.SHARPEN)

        # Step 2: Check if the image is in 'RGB' mode. If not, convert it to 'RGB'.
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Step 3: Convert the PIL image to a numpy array (in RGB format)
        image_array_rgb = np.array(image)

        # Check if the array conversion was successful and the array is not empty
        if image_array_rgb.size != 0:
            # Step 4: Convert the RGB array to BGR format (which is what OpenCV uses)
            image_cv = cv2.cvtColor(image_array_rgb, cv2.COLOR_RGB2BGR)
        else:
            print("The conversion to a NumPy array failed. The array is empty.")

        # Convert to gray scale
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

        # Use adaptive thresholding to highlight the regions of interest
        # This method is effective in varying lighting conditions and contrasting background
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # Find contours on the thresholded image
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area to remove noise
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 30000 and cv2.contourArea(cnt) < 100000]
        # print(len(filtered_contours))
        sc_logger.info(f"No of boxes in this page>>> {len(filtered_contours)}")

        bounding_boxes = []

        # For each contour, find the bounding rectangle and store it
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, x+w, y+h))

            # Draw a rectangle around the contour for visualization
            cv2.rectangle(image_cv, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Save the image with outlined regions of interest
        # generate random 10 digits number
        random_number = random.randint(1000000000, 9999999999)
        
        outlined_image_path = os.path.join(data_dir, 'pdf_images', 'outlined_images', f"""outlined_image_{str(random_number)}.png""")
        cv2.imwrite(outlined_image_path, image_cv)
        sc_logger.info(f"Outlined image dumped to path>>> {outlined_image_path}")
        areas = [cv2.contourArea(cnt) for cnt in filtered_contours]
        # print(sorted(areas))
        return (image_cv, filtered_contours)

    def ocr_on_roi(self,image, contour, index):
        '''Get the bounding box for the contour'''
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the region of interest
        roi = image[y:y+h, x:x+w]

        # Use Tesseract to do OCR on the image
        text = image_to_string(roi, config='--psm 6', output_type=Output.STRING)

        return text.strip()  # Remove any leading/trailing white space

    def ocr_pdf(self):
        sc_logger.info(f"ocr pdf process started >>>>>>>")
        # Apply the adjusted preprocessing to the extracted images
        preprocessed_images_adjusted = [self.preprocess_image_adjusted(img) for img in self.extract_images_from_pdf()]

        # Re-apply OCR to the preprocessed images and collect the results
        preprocessed_ocr_results_adjusted = []
        for img, contours in preprocessed_images_adjusted:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for i, contour in enumerate(contours):
                text = self.ocr_on_roi(gray_image, contour, i)  # Use grayscale image for OCR
                if text:  # If text was detected, save it with an identifier
                    preprocessed_ocr_results_adjusted.append(text)
        # Return the OCR results after adjusted preprocessing

        sc_logger.info(f"ocr pdf process completed >>>>>>>")

        return preprocessed_ocr_results_adjusted
    

    def parse_voter_info(self,raw_text):
        """
        This function takes the raw text data from the OCR and parses out the voter information.
        It uses regular expressions to identify patterns in the text corresponding to the information we want to extract.
        """
        regex_mappings = {
            "id": r"^\s*(\d+)",  # Extracts the ID assuming it is two digits.
            "voter_id": r"\b([A-Z]+\d+)\b",  # Extracts the voter_id assuming it starts with letters followed by numbers.
            "name": r"Name:\s*([^\n]+)",  # Extracts whatever follows "Name: " until the newline.
            "house_number": r"House Number\s*:\s*([\d/]+)",  # Extracts digits that follow "House Number: ".
            "age": r"Age:\s*(\d+)",  # Extracts digits that follow "Age: ".
            "gender": r"Gender:\s*(\w+)"  # Extracts word that follows "Gender: ".
        }
        regex_parent_spouse_name_only = r"(Father's Name|Mother's Name|Husband's Name)\s*:\s*([^\n]+)"

        # Extracting information
        extracted_info = {}
        for key, regex in regex_mappings.items():
            match = re.search(regex, raw_text)
            if match:
                extracted_info[key] = match.group(1).strip()  # Extract the matching text and trim whitespace
        parent_match = re.search(regex_parent_spouse_name_only, raw_text)
        parent_or_spouse_name_only = parent_match.group(2).strip() if parent_match else None 
        extracted_info['parent_or_spouse_name_only'] = parent_or_spouse_name_only
        return extracted_info
    
    def extract_voters_info(self):
        """
        This function combines the OCR and parsing steps to extract the voter information from the PDF.
        """
        preprocessed_ocr_results_adjusted = self.ocr_pdf()
        # print("voters info>>>", preprocessed_ocr_results_adjusted)
        sc_logger.info("extraction of voters info started using regex >>>>>>>")
        for text in preprocessed_ocr_results_adjusted:
            parsed_voters = self.parse_voter_info(text)
            self.refined_parsed_voters.append(parsed_voters)

        sc_logger.info("extraction of voters info completed >>>>>>>")

        return self.refined_parsed_voters
    
    def validate_voters_info(self):
        """
        This function validates the extracted voter information.
        """
        df = pd.DataFrame(self.refined_parsed_voters)
        # if its not numeric, then it is invalid and we can keep it as null
        df['id'] = pd.to_numeric(df['id'], errors='coerce').astype('Int64')

        # if voter_id is not alphanumeric, if it consists of space, then it is invalid and we can keep it as null
        # df['voter_id'] = df['voter_id'].apply(lambda x: x if x.isalnum() else None)

        # age should be numeric, if not, then it is invalid and we can keep it as null
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df['age'] = df['age'].apply(lambda x: x if (x > 16 and x<120) else None)

        # gender should be MALE| FEMALE else it would be nan
        df['gender'] = df['gender'].apply(lambda x: x if x in ('MALE',"FEMALE") else None)

        self.validated_voters_df = df

        sc_logger.info(f"validated_voters_df shape >>> {df.shape}")
        sc_logger.info(f"validated_voters_df >>> {df.head()}")
        return df

    def save_extracted_voters_info(self):
        """
        This function saves the extracted voter information to a CSV file.
        """
        # Extract the voter information
        df = self.validate_voters_info()

        # extract last file name from self.path and exclude .pdf also 
        file_name = os.path.basename(self.path).split('.')[0]

        # Save the DataFrame to a CSV file
        csv_file_path = os.path.join(output_dir, f"{file_name}_output.csv")
        df.to_csv(csv_file_path, index=False)

        return csv_file_path


def parse_voter_pdf():
    # Path to your PDF file
    pdf_file_path = os.path.join(data_dir, 'electoral_rolls.pdf')

    voter_info_parser = VoterInfoParser(pdf_file_path, first_page=3, last_page=31)

    extracted_text = voter_info_parser.extract_voters_info()

    # print(f"""extracted_text>>> {extracted_text}""")

    # print(f"""voters >>> {voter_info_parser.refined_parsed_voters}""")

    csv_file_path = voter_info_parser.save_extracted_voters_info()

    sc_logger.info(f"""csv_file_path>>> {csv_file_path}""")


if __name__ == '__main__':
    parse_voter_pdf()







