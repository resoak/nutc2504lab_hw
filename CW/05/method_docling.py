from docling.document_converter import DocumentConverter

def convert_with_docling(input_path, output_path):
    converter = DocumentConverter()
    result = converter.convert(input_path)
    
    # 將轉換後的內容導出為 Markdown 格式
    markdown_output = result.document.export_to_markdown()
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)
    print(f"Success: Saved to {output_path}")

if __name__ == "__main__":
    convert_with_docling("example.pdf", "output_docling.md")