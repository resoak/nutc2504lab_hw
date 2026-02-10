import pdfplumber

def convert_with_pdfplumber(input_path, output_path):
    full_text = []
    with pdfplumber.open(input_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(full_text))
    print(f"Success: Saved to {output_path}")

if __name__ == "__main__":
    convert_with_pdfplumber("example.pdf", "output_pdfplumber.md")