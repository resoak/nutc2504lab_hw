import logging
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

def run_rapidocr_task():
    # 1. 配置 RapidOCR 專用的 Pipeline 選項
    pipeline_options = PdfPipelineOptions()
    
    # 啟用 OCR
    pipeline_options.do_ocr = True
    
    # 設定圖片縮放比例 (等同於您之前 VLM 裡的 scale=2.0)
    # 對於表格中的細小文字，提高到 3.0 會更精準
    pipeline_options.images_scale = 3.0 
    
    # 關鍵：設定 RapidOCR 引擎選項，支援繁體中文與英文
    pipeline_options.ocr_options = RapidOcrOptions(
        lang=["ch_tra", "en"]
    )

    # 2. 建立文件轉換器
    # 這裡我們將 PDF 的格式選項綁定到剛剛定義的 pipeline_options
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    # 3. 指定輸入檔案與輸出檔案
    input_filename = "sample_table.pdf"
    output_filename = "output_result.md"
    
    input_path = Path(input_filename)
    if not input_path.exists():
        print(f"錯誤：找不到檔案 {input_filename}")
        return

    print(f"正在使用 RapidOCR 解析 {input_filename}...")

    try:
        # 4. 執行轉換
        result = doc_converter.convert(input_path)
        
        # 5. 取得 Markdown 內容
        md_content = result.document.export_to_markdown()

        # 6. 輸出到螢幕並寫入檔案
        if md_content.strip():
            # 寫入 Markdown 檔案
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            print("\n--- 辨識成功 ---")
            print(md_content)
            print(f"\n[完成] Markdown 內容已輸出至: {output_filename}")
        else:
            print("警告：辨識完成但結果為空，請檢查環境是否已安裝 rapidocr_onnxruntime")

    except Exception as e:
        print(f"發生錯誤: {e}")

if __name__ == "__main__":
    run_rapidocr_task()