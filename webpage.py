import streamlit as st
from PIL import Image
import joblib
# from skimage.feature import graycomatrix, graycoprops
# import numpy as np
# import cv2

# def load_model():
#     return joblib.
# output={
#     Name:None
# }
st.title('PLANT DISEASE DETECTION')
st.divider()
im = st.file_uploader('Upload the image ',type=['jpg','jpeg','png'])

but=st.button('Detect the Disease')
if im and but:
    image=Image.open(im)
    st.image(image)
elif but:
    st.warning('Upload the image')