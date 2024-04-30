import streamlit as st


st.title("Multi-Organ Image Segmentation")
st.subheader("Team 3: Liang Gao & Kanishk Goel")

st.divider()

st.header("Dataset")
st.markdown('**id**: unique identifier for object (case123_day20_slice_0001)')
st.markdown('**class**: the predicted class for the object(large_bowel, small_bowel, stomach')
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/data.png", width=500)
st.markdown('**segmentation**: Run-Length Encoding(RLE)-encoded pixels for the identified object(28094 3 28358 7...). RLE-encoded pixels is method of encoding pixel data using RLE, which is a form of lossless data compression.')

st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/slice.png", width=500)

st.subheader("Data processing")
st.markdown('Create new columns: case, day, slice, width, height, path...')
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/col.png", width=600)
st.markdown('**Shape:**(38496, 11)')

st.divider()

st.header("Exploratory Data Analysis (EDA)")
st.subheader("Figure 1：Sample Image")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/image.png")
st.subheader("Figure 2：Image with Masks")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/mask.png")
st.subheader("Figure 3：")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/count.png", width=600)
st.subheader("Figure 4：")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/percent-img-masks.png", width=600)


st.divider()

st.header("Model 1: U-Net")
st.subheader('Brief Intro')
st.markdown("A convolutional neural network (CNN) architecture which was originally developed for biomedical image segmentation tasks.")
st.markdown("Notable for its efficiency and accuracy in segmenting images where the number of labeled samples is relatively small.")
st.markdown("'U-shaped' structure - Contracting Path (Downsampling) & Expansive Path (Upsampling).")
st.markdown("The concatenation of feature maps from the contracting path with those from the expansive path, which helps in precise localization.")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/unet.png")

st.subheader('Experiment')
########### Add demo 1 ############
st.markdown("***Criterion: DICE Loss***")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/diceLoss.png")

st.markdown("***Criterion: BCE Loss***")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/bceLoss.png")

st.markdown("***Criterion: 0.5*BCE + 0.5*DICE loss***")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/comboLoss.png")

st.subheader('Results')
st.markdown("***Taregt***")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/actual.png")

st.markdown("***Prediction***")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/prediction.png")

st.markdown('***DICE coefficient by Classes***')
table_markdown = """
| Large Bowel | Small Bowel | Stomach |
|----------|----------|----------|
| 111 | 222 | 333 |

"""
st.markdown(table_markdown)

st.divider()

st.header("Model 2: F-CNN")
st.subheader('Brief Intro')
st.markdown("Line 1")
st.markdown("Line 2")
st.markdown("Line 3")
st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/fcnn.png")

st.subheader('Experiment results')
########### Add demo 2 ############
st.markdown("**DICE Loss** =")
st.markdown("**DICE Coefficient** = ")


st.header("Conclusion")
st.markdown("xxxxxx")