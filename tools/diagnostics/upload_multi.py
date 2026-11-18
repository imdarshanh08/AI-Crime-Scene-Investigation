import streamlit as st

st.title("CCTV Upload Diagnostic")

uploaded = st.file_uploader(
    "Choose a video",
    type=["mp4", "avi", "mov", "mkv"],
    accept_multiple_files=True
)

if uploaded:
    st.success(f"{len(uploaded)} file(s) received")

    for video in uploaded:
        st.write("Filename:", video.name)
        st.write("Size:", video.size, "bytes")
else:
    st.info("No video selected yet.")
