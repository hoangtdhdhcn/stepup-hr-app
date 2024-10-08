import os
from PyPDF2 import PdfReader
import pandas as pd
from tqdm import tqdm


class CVsReader:
    
    def __init__(self, cvs_directory_path=None):
        self.cvs_directory_path = cvs_directory_path

    # Method to read CV files from uploaded files
    def read_cv_from_files(self, uploaded_files):
        # Initialize a dictionary to hold the filenames and contents of the CVs
        data = {'CV_Filename': [], 'CV_Content': []}

        # For each uploaded file
        for uploaded_file in tqdm(uploaded_files, desc='Processing Uploaded CVs'):
            # Ensure the file is a PDF
            if uploaded_file.type == "application/pdf":
                # Read the content of the PDF file
                content = self._extract_text_from_pdf(uploaded_file)
                # Add the filename and content to the dictionary
                data['CV_Filename'].append(uploaded_file.name)
                data['CV_Content'].append(content)
            else:
                print(f"File {uploaded_file.name} is not a PDF and will be skipped.")
        
        # Return the data as a DataFrame
        return pd.DataFrame(data)

    # Method to extract text from a PDF file
    def _extract_text_from_pdf(self, uploaded_file):
        # Print the name of the file being processed
        print(f"Extracting text from file: {uploaded_file.name}")

        # Create a PdfReader object
        pdf = PdfReader(uploaded_file)

        # Initialize an empty string to store the extracted text
        text = ''

        # Loop over the pages in the pdf
        for page in range(len(pdf.pages)):
            # Extract text from each page and append it to the text string
            text += pdf.pages[page].extract_text()

        # Return the extracted text
        return text

    # Define a method that reads and cleans CVs from a directory
    def read_cv(self):
        print('---- Executing CVs Content Extraction Process ----')

        # Read the PDFs from the directory and store their content in a DataFrame
        df = self._read_pdfs_content_from_directory(self.cvs_directory_path)

        print('Cleaning CVs Content...')
        df['CV_Content'] = df['CV_Content'].str.replace(r"\n(?:\s*)", "\n", regex=True)

        print('CVs Content Extraction Process Completed!')
        print('----------------------------------------------')
        return df

    # Method to extract text from a PDF file
    def _extract_text_from_pdf(self, pdf_path):

        # Print the name of the file being processed
        print(f"Extracting text from file: {pdf_path}")

        # Create a PdfReader object
        pdf = PdfReader(pdf_path)

        # Initialize an empty string to store the extracted text
        text = ''

        # Loop over the pages in the pdf
        for page in range(len(pdf.pages)):

            # Extract text from each page and append it to the text string
            text += pdf.pages[page].extract_text()

        # Return the extracted text
        return text

    
    # Define a method that reads PDF content from a directory
    def _read_pdfs_content_from_directory(self, directory_path):
        
        # Initialize a dictionary to hold the filenames and contents of the CVs
        data = {'CV_Filename': [], 'CV_Content': []}
        
        # Read all the new files in the directory
        all_cvs = self._read_new_directory_files()
        
        # For each file in the directory
        for filename in tqdm(all_cvs, desc='CVs'):
            # If the file is a PDF
            if filename.endswith('.pdf'):
                # Construct the full file path
                file_path = os.path.join(directory_path, filename)
                try:
                    # Extract the text content from the PDF
                    content = self._extract_text_from_pdf(file_path)
                    # Add the filename to the dictionary
                    data['CV_Filename'].append(filename)
                    # Add the content to the dictionary
                    data['CV_Content'].append(content)
                except Exception as e:
                    # Print the exception if there is an error in reading the file
                    print(f"Error reading file {filename}: {e}")
        # Return the data as a DataFrame
        return pd.DataFrame(data)


    # Define a method that reads and cleans CVs
    def read_cv(self):
        
        # Print a message indicating the start of the CV extraction process
        print('---- Excecuting CVs Content Extraction Process ----')
        
        # Read the PDFs from the directory and store their content in a DataFrame
        df = self._read_pdfs_content_from_directory(self.cvs_directory_path)
        
        # Print a message indicating the start of the CV content cleaning process
        print('Cleaning CVs Content...')
        # Clean the CV content by replacing newline characters and trailing spaces with a single newline character
        df['CV_Content'] = df['CV_Content'].str.replace(r"\n(?:\s*)", "\n", regex=True)

        # Print a message indicating the end of the CV extraction process
        print('CVs Content Extraction Process Completed!')
        print('----------------------------------------------')
        # Return the DataFrame
        return df
