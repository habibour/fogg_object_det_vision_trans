from pypdf import PdfReader

reader = PdfReader("2504.10877v1.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print(text)
