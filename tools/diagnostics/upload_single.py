import streamlit as st
from pathlib import Path

st.title("CCTV Upload Diagnostic")

uploaded = st.file_uploader(
    "Choose an MP4 video",
    type=["mp4", "avi", "mov", "mkv"],
    key="upload_test"
)

if uploaded is not None:
    st.write("Filename:", uploaded.name)
    st.write("File size:", uploaded.size, "bytes")
    st.success("Streamlit received the file.")
    if st.button("Save test file"):
        Path("uploads").mkdir(exist_ok=True)
        Path("uploads/test_" + uploaded.name).write_bytes(uploaded.getvalue())
        st.success("Test file saved.")
else:
    st.info("No file received yet.")
