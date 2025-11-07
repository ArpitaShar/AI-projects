import streamlit as st
import ollama
from PIL import Image
from pdf2image import convert_from_bytes
import io
import base64

# ------------------------
# Page Configuration
# ------------------------
st.set_page_config(
    page_title="LLM OCR",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------
# Header
# ------------------------
st.markdown(
    """# <img src="data:image/png;base64,{}" width="50" style="vertical-align: -12px;"> Vision Transformer OCR""".format(
        base64.b64encode(open("./assets/gemma3.png", "rb").read()).decode()
    ), unsafe_allow_html=True
)
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear 🗑️"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract structured text from images and PDFs using Gemma-3 Vision!</p>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------
# Sidebar File Upload
# ------------------------
with st.sidebar:
    st.header("Upload Image or PDF")
    uploaded_file = st.file_uploader("Choose a file...", type=['png', 'jpg', 'jpeg', 'pdf'])

# ------------------------
# OCR Processing
# ------------------------
if uploaded_file is not None:
    file_type = uploaded_file.type

    if file_type == "application/pdf":
        st.write("📄 PDF detected. Converting pages to images...")
        try:
            pdf_images = convert_from_bytes(uploaded_file.read())
            st.image(pdf_images, caption=[f"Page {i+1}" for i in range(len(pdf_images))])

            if st.button("Extract Text 🔍", type="primary"):
                ocr_result = ""
                with st.spinner("Processing PDF pages..."):
                    for i, image in enumerate(pdf_images):
                        buf = io.BytesIO()
                        image.save(buf, format='PNG')
                        image_bytes = buf.getvalue()

                        response = ollama.chat(
                            model='gemma3:12b',
                            messages=[{
                                'role': 'user',
                                'content': f"Extract text from page {i+1} of the PDF in structured Markdown.",
                                'images': [image_bytes]
                            }]
                        )
                        ocr_result += f"## Page {i+1}\n" + response.message.content + "\n\n"

                    st.session_state['ocr_result'] = ocr_result
        except Exception as e:
            st.error(f"❌ Error processing PDF: {str(e)}")

    else:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image")

            if st.button("Extract Text 🔍", type="primary"):
                with st.spinner("Processing image..."):
                    response = ollama.chat(
                        model='gemma3:12b',
                        messages=[{
                            'role': 'user',
                            'content': """Analyze the text in the provided image. Extract all readable content
                                          and present it in a structured Markdown format with headings, lists, and code blocks as needed.""",
                            'images': [uploaded_file.getvalue()]
                        }]
                    )
                    st.session_state['ocr_result'] = response.message.content
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

# ------------------------
# Display OCR Output
# ------------------------
if 'ocr_result' in st.session_state:
    st.markdown("---")
    st.markdown("### 📄 Extracted Text")
    st.markdown(st.session_state['ocr_result'])
else:
    st.info("Upload an image or PDF and click 'Extract Text' to see the results here.")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.markdown("Made using Gemma-3 Vision Model")
