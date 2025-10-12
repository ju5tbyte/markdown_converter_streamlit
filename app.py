import streamlit as st
import PyPDF2
from pdf2image import convert_from_bytes
from PIL import Image
import io
import os
import base64
import re
import zipfile
from pathlib import Path
from mistralai import Mistral
import google.generativeai as genai
from streamlit_paste_button import paste_image_button

st.set_page_config(page_title="OCR Converter Suite", layout="wide")

# API 키 확인
mistral_api_key = os.getenv("MISTRAL_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    genai.configure(api_key=google_api_key)

# 메인 타이틀
st.title("📝 OCR Converter Suite")
st.markdown("PDF 또는 이미지를 마크다운으로 변환합니다.")
st.markdown("---")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["📄 PDF to Markdown (Gemma)", "📄 PDF to Markdown (Mistral)", "📸 Image to Markdown"])

# ============================================================================
# TAB 1: PDF to Markdown (Gemma)
# ============================================================================
with tab1:
    if not google_api_key:
        st.error("❌ GOOGLE_API_KEY 환경 변수가 설정되어 있지 않습니다.")
    else:
        # 세션 스테이트 초기화
        if 'gemma_preview_page_idx' not in st.session_state:
            st.session_state.gemma_preview_page_idx = 0
        if 'gemma_pdf_bytes_cache' not in st.session_state:
            st.session_state.gemma_pdf_bytes_cache = None
        
        def parse_page_selection_gemma(page_input, total_pages):
            """페이지 선택 문자열을 파싱하여 페이지 번호 리스트 반환"""
            pages = set()
            if not page_input.strip():
                return list(range(total_pages))
            
            try:
                parts = page_input.split(',')
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        start, end = part.split('-')
                        start = int(start.strip()) - 1
                        end = int(end.strip()) - 1
                        pages.update(range(start, end + 1))
                    else:
                        pages.add(int(part.strip()) - 1)
                
                return sorted([p for p in pages if 0 <= p < total_pages])
            except:
                return list(range(total_pages))
        
        def convert_pdf_to_markdown_gemma(pdf_bytes, page_indices, prompt, dpi=300):
            """PDF를 Gemma API를 이용해 마크다운으로 변환"""
            try:
                # PDF를 이미지로 변환
                images = convert_from_bytes(pdf_bytes, dpi=dpi)
                
                # 선택된 페이지만 필터링
                selected_images = [images[i] for i in page_indices if i < len(images)]
                
                model = genai.GenerativeModel('gemma-3-27b-it')
                markdown_results = []
                
                # 각 페이지를 개별적으로 처리
                for idx, img in enumerate(selected_images):
                    with st.spinner(f"📄 페이지 {idx + 1}/{len(selected_images)} 변환 중..."):
                        response = model.generate_content([prompt, img])
                        markdown_results.append(response.text)
                
                # 모든 페이지 결과를 합침
                combined_markdown = "\n\n".join(markdown_results)
                return combined_markdown
                
            except Exception as e:
                st.error(f"❌ 변환 중 오류 발생: {str(e)}")
                return None
        
        # 레이아웃
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload & Settings")
            
            uploaded_file_gemma = st.file_uploader("Upload PDF", type=['pdf'], key="pdf_uploader_gemma")
            
            if uploaded_file_gemma is not None:
                pdf_bytes_gemma = uploaded_file_gemma.read()
                st.session_state.gemma_pdf_bytes_cache = pdf_bytes_gemma
                uploaded_file_gemma.seek(0)
            
            dpi_gemma = st.selectbox(
                "DPI (Resolution)",
                options=[200, 300, 400],
                index=1,
                help="이미지 해상도 (높을수록 OCR 정확도 향상, 처리 시간 증가)",
                key="dpi_select_gemma"
            )
            
            page_selection_gemma = st.text_input(
                "Page Selection", 
                placeholder="예: 1, 3-5, 8 (비워두면 전체)",
                help="추출할 페이지를 입력하세요. 예: 1, 3-5, 8",
                key="page_selection_input_gemma"
            )
            
            prompt_gemma = st.text_area(
                "변환 프롬프트를 입력하세요 (기본값 사용 가능):",
                value="""You are a specialized AI assistant with expertise in parsing academic materials for Mathematics and Computer Science. Your mission is to accurately convert the provided image into a structured Markdown document.

Follow these rules strictly:

1. Extract all text content accurately.
2. Convert ALL mathematical equations and formulas to LaTeX format using $ for inline math and $ for display math.
3. For complex mathematical expressions, use LaTeX notation strictly. 
4. Preserve the document structure (headings, lists, tables, etc.). Use `**bold**` for bolded text and `*italic*` for italicized text.
5. Use proper markdown syntax.
6. All code snippets, pseudocode, or terminal commands must be enclosed in triple backticks (```). If you can identify the programming language, specify it (e.g., ```python, ```c++, ```java). Short inline codes must be enclosed in one backticks (`).
7. Bulleted lists must start with a hyphen (`- `). Numbered lists should use numbers (`1. `, `2. `).
8. If the slide contains diagrams, charts, or complex images that cannot be represented as text, describe them briefly in brackets. For example: [Image: Graph showing the process of gradient descent]
9. Get rid of headers or footers such as lecture name, professor, laboratory name.
10. Do not add any explanations, just output the markdown content. (text itself, not the code snippet of markdown)""",
                height=250,
                key="pdf_prompt_gemma"
            )
            
            convert_btn_gemma = st.button("🔄 Convert", type="primary", use_container_width=True, key="pdf_convert_btn_gemma")
            
            if st.session_state.gemma_pdf_bytes_cache is not None:
                st.subheader("📖 PDF Preview")
                try:
                    pdf_bytes = st.session_state.gemma_pdf_bytes_cache
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                    total_pages = len(pdf_reader.pages)
                    
                    selected_pages = parse_page_selection_gemma(page_selection_gemma, total_pages)
                    
                    if len(selected_pages) == 0:
                        st.warning("⚠️ 선택된 페이지가 없습니다.")
                    else:
                        if st.session_state.gemma_preview_page_idx >= len(selected_pages):
                            st.session_state.gemma_preview_page_idx = 0
                        
                        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
                        
                        with col_nav1:
                            if st.button("⬅️ Previous", use_container_width=True, disabled=(st.session_state.gemma_preview_page_idx == 0), key="prev_btn_gemma"):
                                st.session_state.gemma_preview_page_idx = max(0, st.session_state.gemma_preview_page_idx - 1)
                                st.rerun()
                        
                        with col_nav2:
                            st.markdown(f"<div style='text-align: center; padding: 8px;'><b>Page {st.session_state.gemma_preview_page_idx + 1} of {len(selected_pages)}</b><br/>(Original: Page {selected_pages[st.session_state.gemma_preview_page_idx] + 1})</div>", unsafe_allow_html=True)
                        
                        with col_nav3:
                            if st.button("Next ➡️", use_container_width=True, disabled=(st.session_state.gemma_preview_page_idx >= len(selected_pages) - 1), key="next_btn_gemma"):
                                st.session_state.gemma_preview_page_idx = min(len(selected_pages) - 1, st.session_state.gemma_preview_page_idx + 1)
                                st.rerun()
                        
                        current_page_idx = selected_pages[st.session_state.gemma_preview_page_idx]
                        images = convert_from_bytes(
                            pdf_bytes, 
                            dpi=150, 
                            first_page=current_page_idx + 1, 
                            last_page=current_page_idx + 1
                        )
                        
                        if images:
                            st.image(images[0], use_container_width=True)
                        
                        st.info(f"📄 Total Pages: {total_pages} | Selected: {len(selected_pages)} pages")
                    
                except Exception as e:
                    st.error(f"Error previewing PDF: {str(e)}")
        
        with col2:
            st.subheader("📝 Markdown Output")
            
            if convert_btn_gemma and st.session_state.gemma_pdf_bytes_cache is not None:
                try:
                    pdf_bytes = st.session_state.gemma_pdf_bytes_cache
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                    total_pages = len(pdf_reader.pages)
                    
                    page_indices = parse_page_selection_gemma(page_selection_gemma, total_pages)
                    st.info(f"Selected pages: {[p+1 for p in page_indices]}")
                    
                    markdown_result = convert_pdf_to_markdown_gemma(
                        pdf_bytes, 
                        page_indices, 
                        prompt_gemma, 
                        dpi=dpi_gemma
                    )
                    
                    if markdown_result:
                        st.success("✅ Conversion completed!")
                        
                        with st.expander("📄 Preview", expanded=True):
                            st.markdown(markdown_result)
                        
                        with st.expander("📋 Raw Markdown"):
                            st.code(markdown_result, language='markdown')
                        
                        if uploaded_file_gemma is not None:
                            original_name = Path(uploaded_file_gemma.name).stem
                        else:
                            original_name = "document"
                        
                        output_filename = f"OCR_Gemma_{original_name}.md"
                        
                        st.download_button(
                            label="⬇️ Download Markdown",
                            data=markdown_result,
                            file_name=output_filename,
                            mime="text/markdown",
                            use_container_width=True,
                            key="download_md_gemma"
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
            
            elif convert_btn_gemma and st.session_state.gemma_pdf_bytes_cache is None:
                st.warning("⚠️ Please upload a PDF file first.")

# ============================================================================
# TAB 2: PDF to Markdown (Mistral)
# ============================================================================
with tab2:
    if not mistral_api_key:
        st.error("❌ MISTRAL_API_KEY 환경 변수가 설정되어 있지 않습니다.")
    else:
        # 세션 스테이트 초기화
        if 'preview_page_idx' not in st.session_state:
            st.session_state.preview_page_idx = 0
        if 'pdf_bytes_cache' not in st.session_state:
            st.session_state.pdf_bytes_cache = None
        
        def parse_page_selection(page_input, total_pages):
            """페이지 선택 문자열을 파싱하여 페이지 번호 리스트 반환"""
            pages = set()
            if not page_input.strip():
                return list(range(total_pages))
            
            try:
                parts = page_input.split(',')
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        start, end = part.split('-')
                        start = int(start.strip()) - 1
                        end = int(end.strip()) - 1
                        pages.update(range(start, end + 1))
                    else:
                        pages.add(int(part.strip()) - 1)
                
                return sorted([p for p in pages if 0 <= p < total_pages])
            except:
                return list(range(total_pages))
        
        def extract_pages_from_pdf(pdf_bytes, page_indices):
            """PDF에서 선택된 페이지만 추출"""
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            pdf_writer = PyPDF2.PdfWriter()
            
            for idx in page_indices:
                pdf_writer.add_page(pdf_reader.pages[idx])
            
            output = io.BytesIO()
            pdf_writer.write(output)
            output.seek(0)
            return output.getvalue()
        
        def is_landscape(page):
            """페이지가 가로(landscape) 방향인지 확인"""
            width, height = page.size
            return width > height
        
        def create_nin1_pdf(pdf_bytes, n_in_1=2, dpi=300):
            """메모리 최적화된 PDF N-in-1 변환"""
            try:
                # 먼저 총 페이지 수 확인
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                total_pages = len(pdf_reader.pages)
                
                if total_pages == 0:
                    return None
                
                # 첫 페이지만 로드하여 방향 확인 (메모리 절약)
                first_page_img = convert_from_bytes(
                    pdf_bytes, 
                    dpi=dpi, 
                    first_page=1, 
                    last_page=1
                )[0]
                landscape = is_landscape(first_page_img)
                base_width, base_height = first_page_img.size
                
                combined_images = []
                
                # 페이지를 그룹 단위로 처리 (메모리 절약)
                for i in range(0, total_pages, n_in_1):
                    # 현재 그룹의 페이지만 로드
                    group_start = i + 1
                    group_end = min(i + n_in_1, total_pages)
                    
                    # 필요한 페이지만 메모리에 로드
                    group_images = convert_from_bytes(
                        pdf_bytes, 
                        dpi=dpi, 
                        first_page=group_start, 
                        last_page=group_end
                    )
                    
                    if n_in_1 == 2:
                        img1 = group_images[0]
                        img2 = group_images[1] if len(group_images) > 1 else None
                        
                        if landscape:
                            if img2:
                                width = max(img1.width, img2.width)
                                height = img1.height + img2.height
                                combined = Image.new('RGB', (width, height), 'white')
                                combined.paste(img1, (0, 0))
                                combined.paste(img2, (0, img1.height))
                            else:
                                combined = img1
                        else:
                            if img2:
                                width = img1.width + img2.width
                                height = max(img1.height, img2.height)
                                combined = Image.new('RGB', (width, height), 'white')
                                combined.paste(img1, (0, 0))
                                combined.paste(img2, (img1.width, 0))
                            else:
                                combined = img1
                        
                        combined_images.append(combined)
                        
                        # 메모리 즉시 해제
                        del group_images, img1, img2
                        
                    elif n_in_1 == 4:
                        # 4개의 이미지를 2x2 그리드로 배치
                        imgs = group_images + [None] * (4 - len(group_images))
                        
                        combined_width = base_width * 2
                        combined_height = base_height * 2
                        combined = Image.new('RGB', (combined_width, combined_height), 'white')
                        
                        positions = [(0, 0), (base_width, 0), (0, base_height), (base_width, base_height)]
                        for img, pos in zip(imgs, positions):
                            if img:
                                # 크기가 다를 경우 리사이즈
                                if img.size != (base_width, base_height):
                                    img = img.resize((base_width, base_height), Image.Resampling.LANCZOS)
                                combined.paste(img, pos)
                        
                        combined_images.append(combined)
                        
                        # 메모리 즉시 해제
                        del group_images, imgs
                
                # PDF로 저장
                output = io.BytesIO()
                if combined_images:
                    combined_images[0].save(
                        output, 
                        format='PDF', 
                        save_all=True, 
                        append_images=combined_images[1:],
                        optimize=True  # 파일 크기 최적화
                    )
                
                output.seek(0)
                result = output.getvalue()
                
                # 메모리 정리
                del combined_images
                output.close()
                
                return result
                
            except Exception as e:
                st.error(f"N-in-1 변환 중 오류: {str(e)}")
                return None
        
        def call_mistral_ocr(pdf_bytes):
            """Mistral OCR API 호출"""
            pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')
            client = Mistral(api_key=mistral_api_key)
            
            ocr_response = client.ocr.process(
                model="mistral-ocr-latest",
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_base64}"
                },
                include_image_base64=True
            )
            
            markdown_content = ""
            images_dict = {}
            
            for page in ocr_response.pages:
                markdown_content += page.markdown + "\n\n"
                
                if hasattr(page, 'images') and page.images:
                    for img in page.images:
                        if hasattr(img, 'id') and hasattr(img, 'image_base64'):
                            images_dict[img.id] = img.image_base64
            
            return markdown_content.strip(), images_dict
        
        def create_zip_with_attachments(markdown_content, images_dict, original_filename):
            """마크다운과 이미지를 포함한 ZIP 파일 생성"""
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                md_filename = f"OCR_{original_filename}.md"
                zip_file.writestr(md_filename, markdown_content)
                
                for img_id, img_base64 in images_dict.items():
                    img_base64 = re.sub(r'^data:image/.+;base64,', '', img_base64)
                    img_data = base64.b64decode(img_base64)
                    img_path = f"attachments/{img_id}"
                    zip_file.writestr(img_path, img_data)
            
            zip_buffer.seek(0)
            return zip_buffer.getvalue()
        
        # 레이아웃
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload & Settings")
            
            uploaded_file = st.file_uploader("Upload PDF", type=['pdf'], key="pdf_uploader")
            
            if uploaded_file is not None:
                pdf_bytes = uploaded_file.read()
                st.session_state.pdf_bytes_cache = pdf_bytes
                uploaded_file.seek(0)
            
            col_a, col_b = st.columns(2)
            with col_a:
                n_in_1 = st.selectbox(
                    "Pages per sheet",
                    options=[1, 2, 4],
                    index=1,
                    help="한 페이지에 합칠 원본 페이지 수 (비용 절감 옵션)",
                    key="n_in_1_select"
                )
            
            with col_b:
                dpi = st.selectbox(
                    "DPI (Resolution)",
                    options=[200, 300, 400],
                    index=1,
                    help="이미지 해상도 (높을수록 OCR 정확도 향상, 처리 시간 증가)",
                    key="dpi_select"
                )
            
            page_selection = st.text_input(
                "Page Selection", 
                placeholder="예: 1, 3-5, 8 (비워두면 전체)",
                help="추출할 페이지를 입력하세요. 예: 1, 3-5, 8",
                key="page_selection_input"
            )
            
            convert_btn = st.button("🔄 Convert", type="primary", use_container_width=True, key="pdf_convert_btn")
            
            if st.session_state.pdf_bytes_cache is not None:
                st.subheader("📖 PDF Preview")
                try:
                    pdf_bytes = st.session_state.pdf_bytes_cache
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                    total_pages = len(pdf_reader.pages)
                    
                    selected_pages = parse_page_selection(page_selection, total_pages)
                    
                    if len(selected_pages) == 0:
                        st.warning("⚠️ 선택된 페이지가 없습니다.")
                    else:
                        if st.session_state.preview_page_idx >= len(selected_pages):
                            st.session_state.preview_page_idx = 0
                        
                        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
                        
                        with col_nav1:
                            if st.button("⬅️ Previous", use_container_width=True, disabled=(st.session_state.preview_page_idx == 0), key="prev_btn"):
                                st.session_state.preview_page_idx = max(0, st.session_state.preview_page_idx - 1)
                                st.rerun()
                        
                        with col_nav2:
                            st.markdown(f"<div style='text-align: center; padding: 8px;'><b>Page {st.session_state.preview_page_idx + 1} of {len(selected_pages)}</b><br/>(Original: Page {selected_pages[st.session_state.preview_page_idx] + 1})</div>", unsafe_allow_html=True)
                        
                        with col_nav3:
                            if st.button("Next ➡️", use_container_width=True, disabled=(st.session_state.preview_page_idx >= len(selected_pages) - 1), key="next_btn"):
                                st.session_state.preview_page_idx = min(len(selected_pages) - 1, st.session_state.preview_page_idx + 1)
                                st.rerun()
                        
                        current_page_idx = selected_pages[st.session_state.preview_page_idx]
                        images = convert_from_bytes(
                            pdf_bytes, 
                            dpi=150, 
                            first_page=current_page_idx + 1, 
                            last_page=current_page_idx + 1
                        )
                        
                        if images:
                            st.image(images[0], use_container_width=True)
                        
                        st.info(f"📄 Total Pages: {total_pages} | Selected: {len(selected_pages)} pages")
                        
                        if n_in_1 > 1:
                            estimated_pages = (len(selected_pages) + n_in_1 - 1) // n_in_1
                            savings = ((len(selected_pages) - estimated_pages) / len(selected_pages) * 100) if len(selected_pages) > 0 else 0
                            st.success(f"💰 {n_in_1}-in-1: {len(selected_pages)} pages → ~{estimated_pages} pages (약 {savings:.0f}% 절감)")
                    
                except Exception as e:
                    st.error(f"Error previewing PDF: {str(e)}")
        
        with col2:
            st.subheader("📝 Markdown Output")
            
            if convert_btn and st.session_state.pdf_bytes_cache is not None:
                try:
                    with st.spinner("Processing..."):
                        pdf_bytes = st.session_state.pdf_bytes_cache
                        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                        total_pages = len(pdf_reader.pages)
                        
                        page_indices = parse_page_selection(page_selection, total_pages)
                        st.info(f"Selected pages: {[p+1 for p in page_indices]}")
                        
                        extracted_pdf = extract_pages_from_pdf(pdf_bytes, page_indices)
                        
                        if n_in_1 == 1:
                            combined_pdf = extracted_pdf
                            st.info("No page combination (1-in-1)")
                        else:
                            with st.spinner(f"Creating {n_in_1}-in-1 PDF with {dpi} DPI..."):
                                combined_pdf = create_nin1_pdf(extracted_pdf, n_in_1=n_in_1, dpi=dpi)
                        
                        with st.spinner("Converting to Markdown with OCR..."):
                            markdown_result, images_dict = call_mistral_ocr(combined_pdf)
                        
                        st.success("✅ Conversion completed!")
                        
                        has_images = len(images_dict) > 0
                        if has_images:
                            st.info(f"🖼️ Found {len(images_dict)} image(s) in the document")
                        
                        with st.expander("📄 Preview", expanded=True):
                            if has_images:
                                display_markdown = markdown_result
                                for img_id, img_base64 in images_dict.items():
                                    img_markdown = f"![{img_id}]({img_id})"
                                    img_html = f'<img src="{img_base64}" style="max-width: 100%;" />'
                                    display_markdown = display_markdown.replace(img_markdown, img_html)
                                st.markdown(display_markdown, unsafe_allow_html=True)
                            else:
                                st.markdown(markdown_result)
                        
                        with st.expander("📋 Raw Markdown"):
                            st.code(markdown_result, language='markdown')
                        
                        if uploaded_file is not None:
                            original_name = Path(uploaded_file.name).stem
                        else:
                            original_name = "document"
                        
                        if has_images:
                            zip_data = create_zip_with_attachments(markdown_result, images_dict, original_name)
                            output_filename = f"OCR_{original_name}.zip"
                            
                            st.download_button(
                                label="⬇️ Download ZIP (Markdown + Images)",
                                data=zip_data,
                                file_name=output_filename,
                                mime="application/zip",
                                use_container_width=True,
                                key="download_zip"
                            )
                        else:
                            output_filename = f"OCR_{original_name}.md"
                            
                            st.download_button(
                                label="⬇️ Download Markdown",
                                data=markdown_result,
                                file_name=output_filename,
                                mime="text/markdown",
                                use_container_width=True,
                                key="download_md"
                            )
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
            
            elif convert_btn and st.session_state.pdf_bytes_cache is None:
                st.warning("⚠️ Please upload a PDF file first.")

# ============================================================================
# TAB 3: Image to Markdown
# ============================================================================
with tab3:
    if not google_api_key:
        st.error("❌ GOOGLE_API_KEY 환경 변수가 설정되어 있지 않습니다.")
    else:
        # 세션 상태 초기화
        if "image" not in st.session_state:
            st.session_state.image = None
        if "markdown_result" not in st.session_state:
            st.session_state.markdown_result = None
        
        def convert_image_to_markdown(image, prompt):
            try:
                model = genai.GenerativeModel('gemma-3-27b-it')
                
                with st.spinner("🔄 이미지를 변환 중입니다..."):
                    response = model.generate_content([prompt, image])
                    result = response.text
                    return result
                    
            except Exception as e:
                st.error(f"❌ 변환 중 오류 발생: {str(e)}")
                return None
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 이미지 입력")
            
            tab_img1, tab_img2 = st.tabs(["파일 업로드", "클립보드 붙여넣기"])
            
            with tab_img1:
                uploaded_img_file = st.file_uploader(
                    "이미지 파일을 선택하세요",
                    type=["png", "jpg", "jpeg", "webp"],
                    key="image_file_uploader"
                )
                
                if uploaded_img_file is not None:
                    st.session_state.image = Image.open(uploaded_img_file)
            
            with tab_img2:
                st.markdown("**클립보드에서 이미지 붙여넣기 (Ctrl+V)**")
                
                paste_result = paste_image_button(
                    label="📋 클립보드에서 이미지 붙여넣기",
                    key="paste_button",
                )
                
                if paste_result.image_data is not None:
                    st.session_state.image = paste_result.image_data
            
            prompt = st.text_area(
                "변환 프롬프트를 입력하세요 (기본값 사용 가능):",
                value="""Convert this image to markdown format with the following requirements:
1. Extract all text content accurately
2. Convert ALL mathematical equations and formulas to LaTeX format using $ for inline math and $ for display math
3. Preserve the document structure (headings, lists, tables, etc.)
4. Use proper markdown syntax
5. For complex mathematical expressions, use LaTeX notation strictly
6. Do not add any explanations, just output only the markdown text (not code snippet, tex itself) without any additional comments.""",
                height=200,
                key="image_prompt"
            )
            
            if st.session_state.image is not None:
                st.markdown("### 🖼️ 이미지 미리보기")
                st.image(st.session_state.image, use_container_width=True)
                
                if st.button("🚀 마크다운으로 변환", type="primary", use_container_width=True, key="image_convert_btn"):
                    st.session_state.markdown_result = convert_image_to_markdown(st.session_state.image, prompt)
        
        with col2:
            st.subheader("📝 변환 결과")
            
            if st.session_state.markdown_result:
                st.markdown("### 📄 마크다운 미리보기")
                st.markdown(st.session_state.markdown_result)
                
                st.markdown("---")
                
                st.markdown("### 📋 원본 마크다운 텍스트")
                st.code(st.session_state.markdown_result, language="markdown")
            else:
                st.info("👈 왼쪽에서 이미지를 선택하고 변환 버튼을 클릭하세요.")

# 사이드바
with st.sidebar:
    st.markdown("## ℹ️ 사용 방법")
    
    st.markdown("### 📄 PDF to Markdown (Mistral)")
    st.markdown("""
    1. PDF 파일 업로드
    2. 페이지 선택 및 설정 조정
    3. 변환 버튼 클릭
    4. 결과 다운로드
    """)
    
    st.markdown("### 📄 PDF to Markdown (Gemma)")
    st.markdown("""
    1. PDF 파일 업로드
    2. 페이지 선택 및 DPI 설정
    3. 프롬프트 조정 (선택사항)
    4. 변환 버튼 클릭
    5. 결과 다운로드
    """)
    
    st.markdown("### 📸 Image to Markdown")
    st.markdown("""
    1. 이미지 업로드 또는 붙여넣기
    2. 변환 버튼 클릭
    3. 결과 확인 및 복사
    """)
    
    st.markdown("---")
    st.markdown("**PDF OCR Models:**")
    st.markdown("- Mistral OCR (Tab 1)")
    st.markdown("- Gemma 3 27B IT (Tab 2)")
    st.markdown("**Image OCR Model:** Gemma 3 27B IT")