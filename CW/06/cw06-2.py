import logging
import os
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# 1. 定義輔助函數來配置 ApiVlmOptions
def get_olmocr2_vlm_options(
    model: str = "allenai/olmOCR-2-7B-1025-FP8",
    endpoint: str = "https://ws-01.wade0426.me/v1",
    prompt: str = "Convert this page to markdown.",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    api_key: str = "",
) -> ApiVlmOptions:

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    return ApiVlmOptions(
        url=f"{endpoint}/chat/completions",
        params=dict(
            model=model,
            max_tokens=max_tokens,
        ),
        headers=headers,
        prompt=prompt,
        timeout=180,        
        scale=2.0,          
        temperature=temperature,
        response_format=ResponseFormat.MARKDOWN,
    )

# 2. 主要轉換邏輯
def main():
    # 配置 Pipeline 選項
    vlm_pipeline_options = VlmPipelineOptions()
    vlm_pipeline_options.enable_remote_services = True  
    
    vlm_pipeline_options.vlm_options = get_olmocr2_vlm_options(
        model="allenai/olmOCR-2-7B-1025-FP8",
        endpoint="https://ws-01.wade0426.me/v1"
    )

    # 3. 建立 DocumentConverter 並指定使用 VlmPipeline
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=vlm_pipeline_options,
                pipeline_cls=VlmPipeline,  
            )
        }
    )

    # 4. 執行轉換
    input_file_path = Path("sample_table.pdf")  # 請確保檔案存在
    output_dir = Path("output")
    
    # 確保輸出目錄存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定義輸出的 Markdown 檔名
    output_file_path = output_dir / f"{input_file_path.stem}.md"

    print(f"[*] 正在處理檔案: {input_file_path}")
    
    try:
        # 執行轉換
        result = doc_converter.convert(input_file_path)
        
        # 5. 匯出 Markdown 內容
        markdown_output = result.document.export_to_markdown()
        
        # 6. 寫入檔案
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_output)
            
        print(f"--- 轉換成功 ---")
        print(f"[*] 結果已儲存至: {output_file_path}")
        
    except Exception as e:
        print(f"❌ 轉換過程中發生錯誤: {e}")

if __name__ == "__main__":
    main()