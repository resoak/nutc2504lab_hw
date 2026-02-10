import os
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

def run_rapidocr_to_file():
    # 1. 配置 RapidOCR 與 Pipeline 選項
    pipeline_options = PdfPipelineOptions()
    
    # 強制開啟 OCR 辨識
    pipeline_options.do_ocr = True
    
    # 【關鍵修正】針對 sample_table.pdf 的表格數據與繁體中文，提高解析度倍率至 3.0
    # 這樣可以強迫 Docling 渲染出更清晰的影像，解決「辨識結果為空」的問題
    pipeline_options.images_scale = 3.0 
    
    # 設定 RapidOCR 引擎，並指定支援繁體中文與英文
    pipeline_options.ocr_options = RapidOcrOptions(
        lang=["ch_tra", "en"]
    )

    # 2. 建立轉換器，並綁定 PDF 設定
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # 3. 指定輸入與輸出路徑 (假設都在目前資料夾)
    input_filename = "sample_table.pdf"
    output_filename = "output_result.md"
    
    print(f"--- 啟動 RapidOCR 強制辨識模式 ---")
    print(f"正在處理: {input_filename}")

    try:
        # 4. 執行轉換
        result = doc_converter.convert(input_filename)
        
        # 5. 匯出為 Markdown 內容
        md_content = result.document.export_to_markdown()

        # 6. 將結果寫入檔案並輸出至螢幕
        if md_content.strip():
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(md_content)
            
            print("\n--- 辨識成功，Markdown 預覽 ---")
            print(md_content[:500] + "...") # 預覽前 500 字
            print(f"\n[完成] 詳細結果已儲存至: {os.path.abspath(output_filename)}")
        else:
            print("\n[警告] 辨識完成但內容為空。")
            print("建議檢查環境中是否已安裝：pip install rapidocr_onnxruntime")

    except Exception as e:
        print(f"\n[錯誤] 執行過程中發生異常: {e}")

if __name__ == "__main__":
    run_rapidocr_to_file()