import pdfplumber 

with pdfplumber.open("example_resume.pdf") as pdf: 

    first_page = pdf.pages[0]

    text_first_page = first_page.extract_text()
    print(text_first_page)

