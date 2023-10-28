
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


 
