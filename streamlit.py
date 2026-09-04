from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# PAGE CONFIG

st.set_page_config(
    page_title="FashionVision AI",
    page_icon="👕",
    layout="wide"
)

# CLASS NAMES

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# DEVICE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PATHS

BASE_DIR = Path(__file__).resolve().parent

TEST_CSV = (BASE_DIR/"data"/"fashion-mnist_test.csv")

FEATURE_DIR = (BASE_DIR/"saved_features")

FEATURE_DIR.mkdir(parents=True,exist_ok=True)

FEATURE_PATH = (FEATURE_DIR/ "efficientnet_test_features.pt")

# IMAGE TRANSFORM

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# LOAD TEST DATA

@st.cache_data
def load_test_data():

    if not TEST_CSV.exists():

        raise FileNotFoundError(f"Dataset not found: {TEST_CSV}")

    return pd.read_csv(TEST_CSV)

# ROW TO IMAGE

def row_to_image(row):

    pixels = (row.iloc[1:].to_numpy(dtype=np.uint8).reshape(28, 28))
    image = Image.fromarray(pixels)
    return image

# CUSTOM DATASET

class FashionDataset(Dataset):

    def __init__(self,dataframe,transform):

        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self,idx):

        row = self.dataframe.iloc[idx]
        label = int(row.iloc[0])

        image = row_to_image(row)

        # Grayscale -> RGB
        image = image.convert("RGB")
        image = self.transform(image)

        return image, label

# LOAD EFFICIENTNET FEATURE EXTRACTOR

@st.cache_resource
def load_feature_extractor():

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    # Remove classifier
    model.classifier = nn.Identity()

    model = model.to(DEVICE)
    model.eval()

    return model

# LOAD OR EXTRACT TEST FEATURES

def load_or_extract_test_features():

    # LOAD SAVED FEATURES

    if FEATURE_PATH.exists():

        saved = torch.load(FEATURE_PATH,map_location="cpu",weights_only=False)

        return (saved["features"],saved["labels"])

    # EXTRACT FEATURES FIRST TIME

    test_df = load_test_data()

    test_dataset = FashionDataset(test_df,IMAGE_TRANSFORM)

    test_loader = DataLoader(test_dataset,batch_size=64,shuffle=False,num_workers=0,pin_memory=True)

    model = load_feature_extractor()

    all_features = []
    all_labels = []

    progress_bar = st.progress(0,text="Preparing image features...")


    with torch.no_grad():

        for batch_number, (images,labels) in enumerate(test_loader):

            images = images.to( DEVICE )
            features = model(images)

            all_features.append(features.cpu())

            all_labels.append(labels.cpu())

            progress = (batch_number + 1 ) / len(test_loader)

            progress_bar.progress(progress, text="Preparing image features..." )

    progress_bar.empty()

    features = torch.cat(all_features,dim=0 )

    labels = torch.cat(all_labels,dim=0)

    # Save extracted features

    torch.save({ "features": features,  "labels": labels},FEATURE_PATH )

    return features, labels

# RECOMMEND SIMILAR DATASET IMAGES

def recommend_similar_images(features,query_index,top_k):

    # Normalize all features
    normalized_features = F.normalize(features.float(),p=2,dim=1)
    # Query feature
    query_feature = (normalized_features[query_index])

    # Cosine similarity
    similarity_scores = (normalized_features@ query_feature)

    # Remove original query image
    similarity_scores[query_index] = -float("inf")
    # Top similar images
    top_scores, top_indices = torch.topk(similarity_scores,k=top_k)

    return (top_indices.tolist(),top_scores.tolist())


# PREPROCESS UPLOADED IMAGE

def preprocess_uploaded_image(image):

    # Fashion-MNIST style grayscale
    image = image.convert("L")

    # Convert to RGB for EfficientNet
    image = image.convert("RGB")

    image_tensor = IMAGE_TRANSFORM(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

# EXTRACT FEATURE FROM UPLOADED IMAGE

def extract_uploaded_feature(image):

    image_tensor = (preprocess_uploaded_image(image))
    image_tensor = image_tensor.to(DEVICE)
    model = load_feature_extractor()
    model.eval()

    with torch.no_grad():
        feature = model(image_tensor)
    return feature.cpu()

# FIND SIMILAR ITEMS FOR UPLOADED IMAGE

def recommend_uploaded_image(query_feature,test_features,top_k):

    query_feature = F.normalize(query_feature.float(),p=2,dim=1)

    test_features = F.normalize(test_features.float(),p=2,dim=1)

    similarity_scores = (test_features@ query_feature.squeeze(0))

    top_scores, top_indices = torch.topk(similarity_scores,k=top_k)

    return (top_indices.tolist(),top_scores.tolist())

# SIDEBAR

with st.sidebar:

    st.title( "👕 FashionVision AI")
    st.caption("Fashion Image Recommendation")

    page = st.radio("Navigation",
        [
            "🏠 Project Overview",
            "🔎 Recommendation System",
            "🖼️ Images of Similar Items",
            "📊 Model Comparison"
        ]
    )

    st.divider()

# PAGE 1
# PROJECT OVERVIEW

if page == "🏠 Project Overview":

    st.title("👕 Fashion Image Recommendation System")

    st.subheader("Recommendation Systems Engine for Images ""using CNN Architectures")

    st.write(
        """
        This project develops a deep-learning based
        fashion image recommendation system using
        the Fashion-MNIST dataset.

        The system uses pretrained convolutional neural
        networks to understand visual information from
        fashion images and recommend similar fashion items.

        EfficientNet-B0 is used as the main recommendation
        model because of its strong feature extraction
        capability and fast inference performance.
        """
    )

    st.divider()


    # ========================================================
    # KEY FEATURES
    # ========================================================

    st.header("✨ Key Features")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("🔎 Fashion Recommendation")

        st.write(
            """
            Select a Fashion-MNIST image and
            retrieve visually similar fashion items.
            """
        )


    with col2:

        st.subheader("🖼️ Local Image Search")

        st.write(
            """
            Upload an image from your local computer
            and find similar fashion items.
            """
        )


    with col3:

        st.subheader("⚡ Fast Retrieval")

        st.write(
            """
            Extracted image features are saved and
            reused for faster recommendation results.
            """
        )


    st.divider()



    # PROJECT OBJECTIVE

    st.header("🎯 Project Objective")
    st.write(
        """
        The main objective of this project is to develop
        an image-based recommendation engine that can
        identify visually related fashion products.

        This approach can be useful for E-commerce websites,
        online fashion stores, product catalogues and
        visual product search applications.
        """
    )

    st.divider()

    # DATASET

    st.header("📦 Fashion-MNIST Dataset")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Train Images","60,000")

    c2.metric("Training","48,000")

    c3.metric("Validation","12,000")

    c4.metric("Test Images","10,000")

    st.write(
        """
        Fashion-MNIST contains grayscale fashion images
        belonging to 10 different clothing categories.
        """
    )


    st.write(
        """
        **Classes:** T-shirt/top, Trouser, Pullover,
        Dress, Coat, Sandal, Shirt, Sneaker,
        Bag and Ankle boot.
        """
    )

    st.divider()

    # TECHNOLOGIES


    st.header("🛠️ Technology Stack")

    st.write(
        """
        **Python • PyTorch • Torchvision • EfficientNet-B0 •
        Pandas • NumPy • Scikit-learn • PIL • Streamlit**
        """
    )


# PAGE 2
# RECOMMENDATION SYSTEM

elif page == "🔎 Recommendation System":

    st.title("🔎 Fashion Recommendation System")

    st.write("Select an image from the Fashion-MNIST test dataset.")

    try:
        test_df = load_test_data()
    except Exception as error:

        st.error(str(error))
        st.stop()


    # SETTINGS

    col1, col2 = st.columns(2)

    with col1:

        selected_class_name = st.selectbox("Choose Fashion Class",CLASS_NAMES)

    with col2:

        top_k = st.slider("Number of Recommendations",min_value=3,max_value=10,value=5)

    selected_class = CLASS_NAMES.index(selected_class_name)

    class_indices = test_df.index[test_df.iloc[:, 0] == selected_class].tolist()


    # SAMPLE SELECTION
    if ("selected_query_index" not in st.session_state or st.session_state.get("selected_query_class")!= selected_class):

        st.session_state["selected_query_index"] = class_indices[0]

        st.session_state["selected_query_class"] = selected_class


    if st.button("🔄 Change Sample"):

        st.session_state["selected_query_index"] = int(np.random.choice(class_indices))

    query_index = (st.session_state["selected_query_index"])
    query_row = test_df.iloc[query_index]
    query_label = int(query_row.iloc[0])
    query_image = row_to_image(query_row)

    # DISPLAY SELECTED IMAGE

    left, right = st.columns([1, 3])

    with left:

        st.subheader("Selected Image")

        st.image(query_image.resize((200, 200),Image.Resampling.NEAREST),width=200)

    with right:

        st.subheader("Image Details")

        st.write(f"**Class:** "f"{CLASS_NAMES[query_label]}")

    # FIND SIMILAR IMAGES

    run_button = st.button("🔎 Find Similar Images",type="primary",use_container_width=True)

    if run_button:

        with st.spinner("Finding similar images..."):

            test_features, test_labels = (load_or_extract_test_features())

            top_indices, top_scores = (recommend_similar_images(test_features,query_index,top_k))

        st.divider()
        st.subheader(f"Top {top_k} Similar Images")

        columns = st.columns(len(top_indices))

        for rank, (index,score) in enumerate(zip(top_indices,top_scores)):

            result_row = test_df.iloc[index]
            result_label = int(result_row.iloc[0])

            result_image = row_to_image(result_row)

            with columns[rank]:

                st.image(result_image.resize((150, 150),Image.Resampling.NEAREST),use_container_width=True)

                st.markdown(f"**#{rank + 1} "f"{CLASS_NAMES[result_label]}**")

                st.caption(f"Similarity: "f"{score * 100:.2f}%")

# PAGE 3
# IMAGES OF SIMILAR ITEMS

elif page == "🖼️ Images of Similar Items":
    st.title("🖼️ Images of Similar Items")

    top_k = st.slider("Number of Similar Items", min_value=3, max_value=10, value=5)
    uploaded_file = st.file_uploader("Upload Fashion Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        
        st.subheader("Uploaded Image")
        st.image(uploaded_image, width=350)

        find_button = st.button("🔎 Find Similar Items", type="primary", use_container_width=True)

        if find_button:
            with st.spinner("Finding similar fashion items..."):
                query_feature = extract_uploaded_feature(uploaded_image)
                test_features, test_labels = load_or_extract_test_features()
                test_df = load_test_data()
                top_indices, top_scores = recommend_uploaded_image(query_feature, test_features, top_k)

            st.divider()
            st.subheader(f"Top {top_k} Similar Fashion Items")
            
            columns = st.columns(len(top_indices))

            for rank, (index, score) in enumerate(zip(top_indices, top_scores)):
                result_row = test_df.iloc[index]
                result_label = int(result_row.iloc[0])
                result_image = row_to_image(result_row)

                with columns[rank]:
                    st.image(
                        result_image.resize((160, 160), Image.Resampling.NEAREST), 
                        use_container_width=True
                    )
                    st.markdown(f"**#{rank + 1} {CLASS_NAMES[result_label]}**")
                    st.caption(f"Similarity: {score * 100:.2f}%")

    else:
        st.info("Upload a fashion image to find similar items.")


# PAGE 4
# MODEL COMPARISON

elif page == "📊 Model Comparison":
    st.title("📊 Model Comparison")

    st.write(
        """
        VGG16, ResNet50 and EfficientNet-B0 were evaluated
        using the same 10,000 Fashion-MNIST test images.
        """
    )


    # MODEL RESULTS

    comparison_data = pd.DataFrame({
        "Model": ["VGG16", "ResNet50", "EfficientNet-B0"],
        "Accuracy": [0.9380, 0.9129, 0.9257],
        "Precision": [0.937881, 0.913583, 0.925103],
        "Recall": [0.9380, 0.9129, 0.9257],
        "F1 Score": [0.937830, 0.912480, 0.924865],
        "Inference Time (ms/image)": [7.957723, 3.793755, 1.632750]
    })

    # MAIN METRICS
    c1, c2, c3 = st.columns(3)

    c1.metric("Highest Accuracy", "VGG16", "93.80%")
    c2.metric("Fastest Model", "EfficientNet-B0", "1.63 ms/image")
    c3.metric("EfficientNet Accuracy", "92.57%")

    st.divider()

    # RESULT TABLE

    st.subheader("Model Performance Results")

    display_df = comparison_data.copy()
    display_df["Accuracy"] *= 100
    display_df["Precision"] *= 100
    display_df["Recall"] *= 100
    display_df["F1 Score"] *= 100

    display_df = display_df.rename(columns={
        "Accuracy": "Accuracy (%)",
        "Precision": "Precision (%)",
        "Recall": "Recall (%)",
        "F1 Score": "F1 Score (%)"
    })

    st.dataframe(
        display_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    # ACCURACY CHART

    st.subheader("Accuracy Comparison")

    accuracy_chart = comparison_data[["Model", "Accuracy"]].set_index("Model")
    st.bar_chart(accuracy_chart)

    # INFERENCE CHART

    st.subheader("Inference Time Comparison")

    inference_chart = comparison_data[["Model", "Inference Time (ms/image)"]].set_index("Model")
    st.bar_chart(inference_chart)


    # RESULT SUMMARY

    st.subheader("📌 Result Summary")


    st.write(
        """
        **VGG16** achieved the highest accuracy at
        **93.80%**.

        **EfficientNet-B0** achieved **92.57% accuracy**
        with the fastest inference time of approximately
        **1.63 ms per image**.

        **ResNet50** achieved **91.29% accuracy**.

        EfficientNet-B0 therefore provides a strong
        balance between prediction performance and
        computational efficiency, making it suitable
        for the image recommendation application.
        """
    )