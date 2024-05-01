import streamlit as st
import pandas as pd
from toolbox_demo import *
tabs = st.tabs(["📖 My Slides", "💻 My Demo"])

with tabs[0]:
    st.title("Multi-Organ Image Segmentation")
    st.subheader("Team 3: Liang Gao & Kanishk Goel")

    st.divider()

    st.header('Introduction')
    st.markdown('''
    * A specific application of image segmentation techniques in the field of medical imagings.
    * Identify and isolate multiple organs within medical scans such as CT, MRI, or ultrasound images.
    * **Data Source**: UW-Madison Carbone Cancer Center. MRI scans from actual cancer patients on separate days during radiation treatment.
    ''')


    st.divider()
    st.header("Dataset")
    st.markdown('**id**: unique identifier for object')
    st.markdown('**class**: the predicted class for the object(large_bowel, small_bowel, stomach')

    data = {
        "id": ["case123_day20_slice_0001"] * 3,
        "class": ["large_bowel", "small_bowel", "stomach"]
    }
    df = pd.DataFrame(data)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.table(df)

    st.markdown('**segmentation**: Run-Length Encoding(RLE)-encoded pixels for the identified object(28094 3 28358 7...). Run-length encoding is a basic form of data compression where sequences of the **same data value (runs)** are stored as **a single data value and count**. This method is particularly efficient for images with large areas of **uniform pixels**。')

    st.subheader("Data processing")
    st.markdown('''
    * Create new columns: width, height, path... "slice_0105_266_266_1.50_1.50.png" (115488, 3)
    * Create separate column for class: large bowel, small bowel, stomach. The values are corresponding segmentation(RLE-encoded pixels) (38496, 11).
    * Remove images with no masks (16590, 11)
    ''')


    st.divider()

    st.header("Exploratory Data Analysis (EDA)")
    st.markdown("**Figure 1：Sample Image**")
    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/image.png")
    st.markdown("**Figure 2：Image with Masks**")
    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/mask.png")
    st.markdown("**Figure 3：Statistics**")
    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/figure3.png")

    st.divider()

    st.header("Model 1: U-Net")
    st.subheader('Brief Intro')
    st.markdown("""
    * A convolutional neural network (CNN) architecture which was originally developed for biomedical image segmentation tasks.
    * U-shaped' structure - Contracting Path & Expansive Path.
    * The feature maps from the contracting path and expansive path was concatenated.
    """)
    # st.markdown("Notable for its efficiency and accuracy in segmenting images where the number of labeled samples is relatively small.")

    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/unet.png")

    st.divider()

    st.subheader('Experiment & Results')
    st.markdown('**DICE coefficient**: A statistical tool used to measure the similarity between two sets of data. Compare the **pixel-wise agreement** between a **ground truth** segmentation and a **predicted** segmentation.')
    st.latex(r'''
        \text{Dice} = \frac{2 \times |X \cap Y|}{|X| + |Y|}
    ''')

    st.markdown('**DICE Loss**: Directly considers the overlap between the predicted and true segmentation masks')
    st.latex(r'''
        \text{Dice} = \text{1 - Dice Coefficient}
    ''')

    st.divider()


    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/diceLoss.png")
    st.markdown('***Test - DICE coefficient by Classes***')
    table_html = """
    <div style='text-align: center'>
        <table style='margin-left: auto; margin-right: auto;'>
            <tr>
                <th>Large Bowel</th>
                <th>Small Bowel</th>
                <th>Stomach</th>
            </tr>
            <tr>
                <td>0.90</td>
                <td>0.88</td>
                <td>0.93</td>
            </tr>
        </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("***Test - Ground Truth***")
    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/actual.png")
    st.markdown("***Test - Prediction***")
    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/prediction.png")

    st.divider()


    st.header("Model 2: F-CNN")
    st.subheader('Brief Intro')
    st.markdown(''' 
    * Though UNET is quite popular for Bio-medical image segmentation.
    * FCN has also been used for image segmentation previously. UNET is an extension.
    * Take input of arbitrary size and produce correspondingly-sized output with efficient inference and learning.
    * The model architecture consists of an encoder network and a decoder network.
    ''')

    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/fcnn.png")

    st.divider()
    st.markdown('''* For image classification, we downsize image to output one predicted label.''')
    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/fcnn_1.png")

    st.divider()
    st.markdown('''* Rather we can upsample to calculate the pixel wise output.''')
    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/fcnn_2.png")

    st.divider()
    st.markdown('''* This fusing operation actually is just like the boosting / ensemble technique used in VGGNet, where they add the results by multiple model to make the prediction more accurate.''')
    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/fcnn_3.png")

    st.divider()


    st.subheader('Experiment results')
    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/fcnn_4.png")

    st.markdown(''' * **Train score = 0.5*(dice coeff) + 0.5*(IoU coeff) = 0.69** ''')
    st.image("/home/ubuntu/Deep_Learning-Team3/streamlit/fcnn_5.png")

    st.divider()
    st.header('Conclusion')
    st.markdown('''
    * UNet is able to do image localisation by predicting the image pixel by pixel
    * U-Net combines the strengths of traditional FCNs with additional features that make it more effective for image segmentation tasks.
    * The two models differ in symmetricity of the encoder and decoder portions of the network and the skip connections between them.
    ''')




with tabs[1]:
    st.title('Multi Organ Image Segmentation')

    with st.expander(":open_book: How to Use", expanded=True):
        st.write(
            """
        **This demo is created to automatically segment the stomach and intestines on MRI scans.**
        1. **Click the side button to choose model you want test:**
            - U-Net
            - F-CNN  

        2. **Upload the MRI scans**
            """
        )

        # U-Net
    def get_unet(image):
        model = UNET()
        model.load_state_dict(torch.load('/home/ubuntu/Deep_Learning-Team3/code/model_UNET_dice.pt', map_location=device))
        model.eval()
        pred = []
        images = Variable(image).to(device)
        prediction = model(images)
        prediction = (nn.Sigmoid()(prediction) > 0.5).double()
        pred.append(prediction)
        pred = torch.mean(torch.stack(pred, dim=0), dim=0).cpu().detach()
        fig = plot_image_mask(image, pred)
        return fig


    def plot_batch(imgs, msks, size=3):
        if len(imgs) < size:
            size = len(imgs)
        fig = plt.figure(figsize=(5 * 5, 5))
        for idx in range(size):
            plt.subplot(1, 5, idx + 1)
            img = imgs[idx,].permute((1, 2, 0)).cpu().numpy()
            msk = msks[idx].squeeze(0).permute((1, 2, 0)).cpu().numpy()
            show_img2(img, msk)
        st.pyplot(fig)

    def main():
        # Sidebar to select model type
        model_type = st.sidebar.radio("Select Model Type", ("U-Net", "F-CNN"))

        st.title("Upload Your MRI scan")
        uploaded_file = st.file_uploader("Choose an image...", type=["png, jpeg"])

        if st.button("Get Masks"):
            if model_type == "U-Net":
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                image = demo_image(image)
                IMG = get_unet(image)
                st.pyplot(IMG)

            elif model_type == "F-CNN":
                model = load_model('/home/ubuntu/Deep_Learning-Team3/code/best_model_Kanishk.pt')
                image = Image.open(uploaded_file)
                image = image.convert('RGB')
                img, preds = predict(image, model)
                plot_batch(img, preds, 5)



    if __name__ == "__main__":
        main()