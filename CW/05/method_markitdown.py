from markitdown import MarkItDown

def convert_with_markitdown(input_path, output_path):
    md = MarkItDown()
    result = md.convert(input_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result.text_content)
    print(f"Success: Saved to {output_path}")

if __name__ == "__main__":
    convert_with_markitdown("example.pdf", "output_markitdown.md")