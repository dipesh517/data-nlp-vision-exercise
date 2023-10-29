
# Approach 


Task 1

To accomplish this task, we'll follow these steps:

 1. Extract Text from PDF: 
 
    I tried extracting the text using PDFMiner / PDFPlumber. 
    The text extraction did not work as intended. The output contains a series of characters ('\x0c'), which suggests that the pages might be images or the content is protected or encoded in a way that prevents standard text extraction.

    For this task, especially if the PDF contains images instead of selectable text (common in scanned documents), we would need to use Optical Character Recognition (OCR) to "read" the text from images. A popular library for OCR is Tesseract.

    Tesseract does not directly handle PDFs, but we can convert our PDF pages into images and then use Tesseract to do the OCR on the images.

    Here's a high-level overview of the steps involved if we proceed with this method:

     1. Convert the specific PDF page into an image. We can use a library like pdf2image for this purpose.
     2. Use Tesseract to perform OCR on the image and extract the text.
       
     The OCR process is quite resource-intensive and time-consuming, especially for documents with a significant number of pages or complex layouts.

     Processing the document in smaller batches should help manage the processing time. 


     To improve the parsing accuracy, I have considered the following steps:

        1. Resize the image: Increase the dimensions of the image, which can help in recognizing small text.
        2. Convert to grayscale: This simplification can assist with other transformations.
        3. Increase contrast: This step should occur while the image is in grayscale mode, not binary, to enhance the distinctions between text and background effectively.
        4. Binarization: Convert the image to a binary format, enhancing the distinction between the text (black) and background (white).
        5. Sharpening: This step will make edges more distinct, which can improve text recognition.

 2. Parse the Text: Once we have the text, we'll need to parse it to extract the individual pieces of information. This process might involve looking for patterns in the text, possibly using regular expressions to identify and extract the desired data points.

 3. Organize Data into a Structured Format: After parsing the text, we'll organize the data into a structured format (like a list of dictionaries), where each dictionary corresponds to a voter and contains keys and values for the details we've extracted.

 4. Export to CSV/XLSX: We'll then export this structured data into a CSV or XLSX file, with each row representing a voter's information and each column representing a different detail about the voter.

 5. Validation and Quality Checks: After creating the file, we'll suggest methods to quickly verify the correctness and completeness of the data. Like age should be greater than 16 and less than 120.. name should not be numeric... Gender should be all caps and should be either MALE/FEMALE/OTHERS



Challenges Faced:
Initially I tried extracting the text from whole page. But it was very difficult to find patterns from the extracted text. 
name, relative name, age, gender is easy to extract. But it is not able to extract voter id which is not in a proper pattern. It was not able to find patterns 
So I am trying cv2 to generate image with outlines by detecting boxes or regions of interest (ROIs). And then perform OCR on that particular region of interest.


Using regex to extract voters info from text, it was unable to parse id column, and if names are in multiple lines and if certain patterns changes, the accuracy of parsing the data was quite low. There were duplicates in voter ids also. Few names were not being captured.

To overcome the shortfalls, we used llm for extracting the voter info from the text.
No of duplicates by voter ids decreased too from 31 to 1 case.
Accuracy increased from 91.65 to 99.74 %. 

Summary of data missing present in both datasets :

Regex-parsed data:
voter_id: 31 missing ( Improperly labelled)
name: 33 missing
house_number: 53 missing
parent_or_spouse_name_only: 11 missing

LLM-parsed data:
id: 345 missing Not properly extracted from OCR
voter_id: 1 missing
age: 1 missing
gender: 1 missing
parent_or_spouse_name_only: 1 missing

Detailed Comparative analysis using llm and regex is demonstrated here at #comparative_analysis_using_llm_and_regex.ipynb

Links to relevant code : #src/voter_info_parser.py

python3 -m src.voter_info_parser 

# Output Results
electoral_rolls_output.csv : using regex 
electoral_rolls_parsed_by_llm_output.csv : using llm 

outlined images : data/pdf_images/outlined_images


Test framework:
pytest tests/test_voter_info_parser.py
-------------------------------------------------------------------------------------------------------------------

Task 2 Approach 

Certainly, using a language model, especially a large language model (LLM) like GPT-3 or GPT-4, can be a creative and effective approach to verify the gender classification in our dataset from poll cards.

Steps 

1. Data Preparation:
   CSV file containing voters info generated by parsing the pdf file will be used for gender prediction. 
   We will be using voters name and gender column only.
    
2. Using LLM for Gender Prediction:

    We will use LLM to predict gender based on the names listed in the poll cards, assuming that these are the most indicative data points for gender. The model can analyze the names and use its extensive database of language usage patterns to predict the gender associated with each name.

    We will be using gpt-3.5 turbo 16k LLM Model for this use case.
    Writing a system message specific to the use case.
    Prompt Engineering to create custom prompt that takes csv file and outputs the gender classification.

 3. Cross-Verification:

    After obtaining predictions from the LLM, cross-verify these with the gender classifications listed in your poll cards. Discrepancies might suggest errors in the original dataset.
    Calculate the accuracy by comparing the number of instances where the LLM's prediction matched the gender classification in your data against the total number of entries.

 4. Statistical Analysis:

    Perform statistical analysis to understand the significance of the discrepancies. Are they random, or do they show some pattern? This step will help understand if the mismatches are due to incorrect classifications in the original data or are limitations of the LLM.


 5. Human Review:

    For entries with discrepancies, a human review could be essential. This step involves manually checking a sample of entries to understand the nature of any classification errors and to ensure that the LLM's predictions are interpreted correctly.


LLM Analysis outputs are voters_info_llm_output.csv, llm_results_analysis.csv 
notebook can be referreed for statistical analysis: #llm_analysis.ipynb

Code reference: #src/llm.py 

python3 -m src.llm 

Analysis:
Using llm to verify gender column that is parsed, it showed that the 94 percent of the cases genders matched showing that gender is appropriately parsed by the parser. 
6 % of the cases might be due to inaccuracy of llm to separate genders based on names.

------------------------------------------------------------------------------------------------------------------

Task 3 Approach




