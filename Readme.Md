
# Approach 


Task 1

To accomplish this task, we'll follow these steps:

 1. Extract Text from PDF: 
 
    I tried extracting the text using PDFMiner / PDFPlumber. 
    The text extraction did not work as intended. The output contains a series of characters ('\x0c'), which suggests that the pages might be images or the content is protected or encoded in a way that prevents standard text extraction.

    For this task, especially if the PDF contains images instead of selectable text (common in scanned documents), we would need to use Optical Character Recognition (OCR) to "read" the text from images. A popular library for OCR is Tesseract.
 


 2. Parse the Text: Once we have the text, we'll need to parse it to extract the individual pieces of information. This process might involve looking for patterns in the text, possibly using regular expressions or natural language processing techniques, to identify and extract the desired data points.

 3. Organize Data into a Structured Format: After parsing the text, we'll organize the data into a structured format (like a list of dictionaries), where each dictionary corresponds to a voter and contains keys and values for the details we've extracted.

 4. Export to CSV/XLSX: We'll then export this structured data into a CSV or XLSX file, with each row representing a voter's information and each column representing a different detail about the voter.

 5. Validation and Quality Checks: After creating the file, we'll suggest methods to quickly verify the correctness and completeness of the data.



 
