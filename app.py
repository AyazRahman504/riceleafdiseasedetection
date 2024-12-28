import streamlit as st
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import base64
BASE_DIR = Path(__file__).parent 
# Debugging: Validate paths
# -------------------------------------------------------------------
# 0. PAGE CONFIG & GLOBAL STYLES
# -------------------------------------------------------------------

# Set a wide layout and custom page title
st.set_page_config(page_title="🌾 Rice Leaf Disease Detection", layout="wide")

# Inject custom CSS to style the entire app
st.markdown(
    """
    <style>
    /* Import a modern, clean font (Poppins) */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap');
    
    /* Apply the font to the entire app */
    html, body, [class*="css"]  {
        font-family: 'Poppins', sans-serif;
        background-color: #1E1E1E; /* fallback background if gradient fails */
        color: #FFFFFF;
    }
    
    /* Make the main app area use a subtle gradient background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(120deg, #2F2F2F, #1E1E1E);
        color: #FFFFFF;
        padding: 1rem;
    }

    /* Style the default headings for a modern look */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-weight: 600;
        margin-top: 0.8em;
        margin-bottom: 0.4em;
    }

    /* Adjust paragraph text color slightly lighter for good contrast */
    p, div, ul, li, span, label {
        color: #DDDDDD;
    }

    /* Make the sidebar background a nice dark gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(145deg, #424242, #2C2C2C);
        color: #FFFFFF;
        padding: 1rem;
    }
    
    /* Change all text in the sidebar to white and set the same font family */
    [data-testid="stSidebar"] * {
        color: #FFFFFF;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Adjust spacing & typography for radio buttons in the sidebar */
    div[data-baseweb="radio"] > div {
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
    }
    div[data-baseweb="radio"] label {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
    }
    
    /* Add spacing between the sidebar's title and the radio options */
    section[data-testid="stSidebar"] > div:nth-child(1) > div {
        margin-bottom: 1.5rem;
    }

    /* Give images a subtle drop shadow */
    img {
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.6);
        border-radius: 4px;
    }

    /* Buttons styling */
    button, .stButton button {
        background-color: #4FC3F7 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
        padding: 0.5rem 1rem;
    }
    button:hover, .stButton button:hover {
        background-color: #03A9F4 !important;
    }

    /* Style any text input fields, text areas, etc. */
    input, textarea {
        color: #000000 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #CCCCCC !important;
        border-radius: 4px !important;
        padding: 0.5rem !important;
    }

    /* Subtle horizontal rule style for dividers */
    hr, [data-testid="stMarkdown"] hr {
        border: 0;
        border-top: 1px solid #444444;
        margin: 1.5rem 0;
    }

    /* Gallery Card Styles */
    .gallery-card {
        background-color: #2B2B2B; /* Slightly lighter than main background */
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease-in-out;
        overflow: hidden; /* Prevent overflow on scaling */
    }
    .gallery-card:hover {
        transform: scale(1.02);
    }
    .gallery-image {
        display: block;
        margin: 0 auto;
        border-radius: 4px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.6);
    }
    .gallery-caption {
        text-align: center;
        font-weight: 500;
        color: #DDD;
        margin-top: 8px;
    }

    /* Active Radio Button Styling */
    .css-1a4r52l {
        background-color: #616161 !important;
        border-radius: 8px;
    }

    /* Change cursor on hover */
    div[data-baseweb="radio"] > div:hover {
        background-color: #555555 !important;
        cursor: pointer;
    }

    /* Highlight the selected radio button */
    div[data-baseweb="radio"] > div[aria-checked="true"] {
        background-color: #757575 !important;
    }

    /* Home Page Enhancements */
    .home-header {
        text-align: center;
        padding: 2rem 0;
    }
    .home-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .home-header p {
        font-size: 1.2rem;
        color: #CCCCCC;
    }
    .feature-card {
        background-color: #2B2B2B;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease-in-out;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# 1. HELPER FUNCTIONS
# -------------------------------------------------------------------

@st.cache_resource
def load_model_cached(model_path):
    """
    Load YOLO model with caching to improve performance.
    """
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model at {model_path}: {e}")
        return None

def load_images_from_folder(folder_path):
    """
    Returns a list of absolute paths to all .jpg, .jpeg, or .png files in 'folder_path'.
    """
    valid_exts = ('.jpg', '.jpeg', '.png')
    image_files = []
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(valid_exts):
                image_files.append(os.path.join(folder_path, file_name))
    else:
        st.warning(f"Folder path does not exist: {folder_path}")
    return image_files

def display_images_in_columns(image_paths):
    """
    Displays the images in the provided list 'image_paths' in a 2-column layout,
    with a styled HTML caption below each image.
    """
    num_cols = 2  
    for i in range(0, len(image_paths), num_cols):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            if i + j < len(image_paths):
                with cols[j]:
                    img_path = image_paths[i + j]
                    filename = os.path.basename(img_path)
                    caption_title = os.path.splitext(filename)[0]
                    
                    st.image(img_path, use_container_width=True)
                    
                    caption_html = f"""
                    <p style="
                        text-align: center; 
                        font-size: 1rem; 
                        font-family: 'Poppins', sans-serif;
                        color: #DDDDDD; 
                        margin-top: -10px;
                        margin-bottom: 20px;">
                        <strong>{caption_title}</strong>
                    </p>
                    """
                    st.markdown(caption_html, unsafe_allow_html=True)

def display_disease_info():
    # -- Title of the page --
    st.title("🌾 Rice Leaf Disease Information")
    st.markdown(
        "Below is an overview of common rice leaf diseases, along with links for further reading. "
        "Each section is styled for better readability."
    )
    
    # -- Custom HTML styling (background, margin, etc.) for disease sections --
    html_content = """
    <style>
        .disease-card {
            background-color: #2B2B2B;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
        }
        .disease-title {
            color: #FFFFFF;
            margin: 0;
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
        }
        .disease-icon {
            margin-right: 0.5rem;
        }
        .disease-description {
            line-height: 1.5;
            color: #DDDDDD;
        }
        .disease-link {
            color: #4FC3F7;
            text-decoration: none;
        }
        .disease-link:hover {
            text-decoration: underline;
        }
    </style>
    
    <!-- 1. Bacterial Blight -->
    <div class="disease-card">
        <h3 class="disease-title">
            <span class="disease-icon">🦠</span>1. Bacterial Blight
        </h3>
        <p class="disease-description">
            Bacterial Blight is a major disease affecting rice crops. It causes lesions on leaves, 
            eventually leading to plant death.
        </p>
        <ul>
            <li>
                <a class="disease-link" 
                   href="http://www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-sheets/item/bacterial-blight"
                   target="_blank">
                   Read more about Bacterial Blight
                </a>
            </li>
        </ul>
    </div>

    <!-- 2. Brown Spot -->
    <div class="disease-card">
        <h3 class="disease-title">
            <span class="disease-icon">🍂</span>2. Brown Spot
        </h3>
        <p class="disease-description">
            Brown Spot is caused by a fungal infection, leading to small brown spots on leaves 
            that can kill the entire leaf.
        </p>
        <ul>
            <li>
                <a class="disease-link"
                   href="http://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/brown-spot#:~:text=Brown%20spot%20is%20a%20fungal,can%20kill%20the%20whole%20leaf"
                   target="_blank">
                   Read more about Brown Spot
                </a>
            </li>
        </ul>
    </div>
    
    <!-- 3. Rice Blast -->
    <div class="disease-card">
        <h3 class="disease-title">
            <span class="disease-icon">🍃</span>3. Rice Blast
        </h3>
        <p class="disease-description">
            Rice Blast is a fungal disease that affects the plant’s leaves and can reduce 
            grain yield significantly.
        </p>
        <ul>
            <li>
                <a class="disease-link"
                   href="http://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/blast-leaf-collar#:~:text=Rice%20blast%20is%20one%20of,grain%20fill%2C%20reducing%20grain%20yield"
                   target="_blank">
                   Read more about Rice Blast
                </a>
            </li>
        </ul>
    </div>
    
    <!-- 4. Healthy Rice Leaves -->
    <div class="disease-card">
        <h3 class="disease-title">
            <span class="disease-icon">🍀</span>4. Healthy Rice Leaves
        </h3>
        <p class="disease-description">
            Healthy rice leaves are free from any disease and show no signs of lesions or abnormalities. 
            These leaves are crucial for photosynthesis and optimal plant growth.
        </p>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def predict_and_save(model, img, model_folder):
    try:
        results = model.predict(source=img, conf=0.25, imgsz=640, save=False)
        if results:
            result_image = results[0].plot()
            result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            save_path = os.path.join(model_folder, f"result_{np.random.randint(1e6)}.jpg")
            result_pil.save(save_path)
            return save_path
        else:
            return None
    except FileNotFoundError:
        st.error(f"Model file not found at {model_folder}. Please check the path.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def create_gauge(title, value, max_value=1.0, color="#4FC3F7"):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        gauge = {
            'axis': {'range': [0, max_value]},
            'bar': {'color': color},
            'steps' : [
                {'range': [0, 0.5], 'color': "#FF4C4C"},
                {'range': [0.5, 0.75], 'color': "#FFAA00"},
                {'range': [0.75, max_value], 'color': "#4FC3F7"}],
            'threshold' : {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value}
        }
    ))
    fig.update_layout(height=250, template='plotly_dark')
    return fig

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded as a CSV file."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings
    return f'<a href="data:file/csv;base64,{b64}" download="model_comparison.csv">📥 Download Comparison Data as CSV</a>'

# Dictionary of YOLO model paths
model_paths = {
    "YOLOv8.pt": r"Models/Models/YOLOv8Best.pt",
    "YOLOv10.pt": r"Models/Models/YOLOv10Best.pt",
    "YOLOv11.pt": r"Models/Models/YOLOv11Best.pt",
}

# Folders for storing detection results
yolov8_folder = 'yolov8'
yolov10_folder = 'yolov10'
yolov11_folder = 'yolov11'

# Dictionary of model analytics images folder paths
model_results_paths = {
    "YOLOv8": BASE_DIR / "Models_and_Results" / "Yolo_v8",
    "YOLOv10": BASE_DIR / "Models_and_Results" / "Yolo_v10",
    "YOLOv11": BASE_DIR / "Models_and_Results" / "Yolo_v11",
}
def display_images_from_directory(directory):
    if directory.exists() and directory.is_dir():
        st.write(f"📂 Displaying images from: {directory}")
        
        # List only image files in the directory
        images = list(directory.glob('*.*'))  # Matches all files
        image_files = [f for f in images if f.suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif'}]
        
        if image_files:
            for image_file in image_files:
                try:
                    img = Image.open(image_file)
                    st.image(img, caption=image_file.name, use_column_width=True)
                except Exception as e:
                    st.error(f"Error loading image {image_file.name}: {e}")
        else:
            st.write("No images found in this directory.")
    else:
        st.error(f"Directory {directory} does not exist!")

# Debugging step: Check if paths are correct
st.write(f"Base directory: {BASE_DIR}")
for model, directory in model_results_paths.items():
    st.write(f"{model} path: {directory} - Exists: {directory.exists()}")

# Display images from all specified directories
for model, directory in model_results_paths.items():
    display_images_from_directory(directory)
for model, path in model_results_paths.items():
    st.write(f"📂 {model} Path: {path} - Exists: {os.path.exists(path)}")
    if os.path.exists(path):
        st.write(f"Contents of {model} folder: {os.listdir(path)}")
# Ensure all folders are created once
for folder in [yolov8_folder, yolov10_folder, yolov11_folder]:
    create_folder(folder)

# -------------------------------------------------------------------
# 2. STREAMLIT APPLICATION
# -------------------------------------------------------------------

st.sidebar.title("🌟 Navigation")
page = st.sidebar.radio(
    "Choose a page",
    [
        "🏠 Home", 
        "📖 Disease Information", 
        "📊 Model Analytics", 
        "📷 Image Gallery", 
        "📈 Model Comparison",
        "📂 Results by Model", 
        "📞 Contact Us"
    ]
)

if page == "🏠 Home":
    st.title("🌾 Rice Leaf Disease Detection")
    st.markdown("""
    Welcome to the **Rice Leaf Disease Detection** application. 
    This tool leverages advanced YOLO models to identify and analyze diseases affecting rice crops.
    Upload images of rice leaves to get instant detection results and insights.
    """)
    
    # -------------------------------------------------------------------
    # Home Page Enhancements
    # -------------------------------------------------------------------
    
    # Section: Features
    st.subheader("✨ Key Features")
    cols = st.columns(3)
    features = [
        {"icon": "📸", "title": "Image Upload", "description": "Easily upload images of rice leaves for analysis."},
        {"icon": "🧠", "title": "Advanced Detection", "description": "Utilizes state-of-the-art YOLO models for accurate disease detection."},
        {"icon": "📊", "title": "Performance Analytics", "description": "View detailed analytics and performance metrics of each model."},
    ]
    
    for col, feature in zip(cols, features):
        with col:
            st.markdown(f"<div style='text-align: center;'><span style='font-size: 2rem;'>{feature['icon']}</span></div>", unsafe_allow_html=True)
            st.markdown(f"### {feature['title']}")
            st.markdown(f"{feature['description']}")
    
    st.markdown("---")  # Horizontal divider

    # Section: Image Upload with Drag and Drop
    st.subheader("🔄 Upload Your Own Images")
    st.markdown("Drag and drop images of rice leaves to detect and analyze diseases.")

    # Enhanced File Uploader with Drag and Drop Instructions
    uploaded_images = st.file_uploader(
        "Drag and drop your images here, or click to select files",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload images of rice leaves to detect diseases."
    )

    model_choice = st.selectbox("Select the model for detection", ["YOLOv8.pt", "YOLOv10.pt", "YOLOv11.pt"])
    model = load_model_cached(model_paths[model_choice])
    model_folder = {
        "YOLOv8.pt": yolov8_folder,
        "YOLOv10.pt": yolov10_folder,
        "YOLOv11.pt": yolov11_folder
    }[model_choice]

    if uploaded_images and model:
        st.subheader("🧪 Detection Results")
        for img_file in uploaded_images:
            try:
                image = Image.open(img_file)
                img = np.array(image)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
                result_path = predict_and_save(model, img, model_folder)
                if result_path:
                    st.image(result_path, caption=f"Result for {img_file.name}", use_container_width=True)
                else:
                    st.write(f"🚫 No objects detected in {img_file.name}.")
            except Exception as e:
                st.error(f"Error processing {img_file.name}: {e}")
    elif uploaded_images and not model:
        st.error("Model could not be loaded. Please check the model path and try again.")

elif page == "📖 Disease Information":
    display_disease_info()

elif page == "📊 Model Analytics":
    st.title("📊 Model Analytics")
    st.markdown("Below are the analytics images for each YOLO model.")
    
    st.subheader("YOLOv8 Model Analytics")
    y8_images = load_images_from_folder(model_results_paths["YOLOv8"])
    if not y8_images:
        st.write("🚫 No YOLOv8 analytics images found.")
    else:
        display_images_in_columns(y8_images)
    
    st.markdown("---")  # horizontal divider

    st.subheader("YOLOv10 Model Analytics")
    y10_images = load_images_from_folder(model_results_paths["YOLOv10"])
    if not y10_images:
        st.write("🚫 No YOLOv10 analytics images found.")
    else:
        display_images_in_columns(y10_images)
    
    st.markdown("---")  # horizontal divider

    st.subheader("YOLOv11 Model Analytics")
    y11_images = load_images_from_folder(model_results_paths["YOLOv11"])
    if not y11_images:
        st.write("🚫 No YOLOv11 analytics images found.")
    else:
        display_images_in_columns(y11_images)

elif page == "📷 Image Gallery":
    st.title("📷 Image Gallery")
    st.markdown("Explore images of different rice leaf diseases for reference.")

    from PIL import Image as PILImage
    from PIL import ImageOps

    # Insert inline CSS to style "cards" and hover effects
    st.markdown("""
        <style>
        .gallery-card {
            background-color: #2B2B2B; /* Slightly lighter than main background */
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
            transition: transform 0.2s ease-in-out;
            overflow: hidden; /* so any scaling doesn't overflow the card border */
        }
        /* Scale up a bit when hovering over the card */
        .gallery-card:hover {
            transform: scale(1.02);
        }
        .gallery-image {
            display: block;
            margin: 0 auto;
            border-radius: 4px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.6);
        }
        .gallery-caption {
            text-align: center;
            font-weight: 500;
            color: #DDD;
            margin-top: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # List each image path and caption
    gallery_images = [
        {
            "path": r"Models/Models/3.jpg",
            "title": "Bacterial Blight"
        },
        {
            "path": r"Models/Models/5.jpg",
            "title": "Brown Spot"
        },
        {
            "path": r"Models/Models/1.jpg",
            "title": "Rice Blast"
        },
        {
            "path": r"Models/Models/6.jpg",
            "title": "Healthy Leaf"
        },
    ]

    # Size of your uniform thumbnails (width, height)
    THUMB_SIZE = (300, 300)

    # Helper function to create a uniform-size thumbnail using center-cropping
    def make_uniform_thumbnail(img_path, size=(300,300)):
        try:
            pil_img = PILImage.open(img_path)
            thumb = ImageOps.fit(
                pil_img, 
                size,
                method=PILImage.Resampling.LANCZOS, 
                bleed=0.0, 
                centering=(0.5, 0.5)
            )
            return thumb
        except Exception as e:
            st.error(f"Error loading image {img_path}: {e}")
            return None

    # Number of images per row
    images_per_row = 2  # Changed from 3 to 2 for better arrangement
    
    # Dynamically create rows
    for i in range(0, len(gallery_images), images_per_row):
        row = gallery_images[i : i + images_per_row]
        cols = st.columns(len(row))
        
        # Place each image in its column
        for col, img_info in zip(cols, row):
            uniform_img = make_uniform_thumbnail(img_info["path"], THUMB_SIZE)
            
            # Wrap each image + caption in a "gallery-card" div
            with col:
                st.markdown(
                    "<div class='gallery-card'>",
                    unsafe_allow_html=True
                )
                if uniform_img:
                    st.image(
                        uniform_img,
                        use_container_width=True,
                        caption=None,  # We'll create our own caption below
                        # No more use_column_width to avoid deprecation warnings
                    )
                    st.markdown(
                        f"<p class='gallery-caption'>{img_info['title']}</p>",
                        unsafe_allow_html=True
                    )
                else:
                    st.write("🚫 Image could not be loaded.")
                st.markdown(
                    "</div>",
                    unsafe_allow_html=True
                )

elif page == "📈 Model Comparison":
    st.title("📈 Model Comparison")
    st.markdown("""
    Compare the precision of different YOLO models for specific rice leaf diseases.
    Select two models and a disease to see a detailed comparison.
    Enjoy interactive and playful tools to explore the performance metrics!
    """)

    # Define precision data
    precisions = {
        "YOLOv8.pt": {
            "Brown Spot": 0.89,
            "Healthy Leaf": 0.96,
            "Rice Blast": 0.87,
            "Bacterial Blight": 0.88,
            "All Classes": 0.875  # mAP@0.5 included separately
        },
        "YOLOv10.pt": {
            "Brown Spot": 0.87,
            "Healthy Leaf": 0.94,
            "Rice Blast": 0.85,
            "Bacterial Blight": 0.86,
            "All Classes": 0.881
        },
        "YOLOv11.pt": {
            "Brown Spot": 0.91,
            "Healthy Leaf": 0.98,
            "Rice Blast": 0.90,
            "Bacterial Blight": 0.91,
            "All Classes": 0.925
        }
    }

    # Define mAP@0.5 data for All Classes
    map_values = {
        "YOLOv8.pt": 0.875,
        "YOLOv10.pt": 0.881,
        "YOLOv11.pt": 0.925
    }

    # Model selection
    model_options = list(model_paths.keys())
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Select First Model", model_options, key="model1")
    with col2:
        model2 = st.selectbox("Select Second Model", model_options, index=1, key="model2")

    # Disease selection
    disease_options = ["Brown Spot", "Healthy Leaf", "Rice Blast", "Bacterial Blight", "All Classes"]
    disease = st.selectbox("Select Disease for Comparison", disease_options)

    # Validate model selections
    if model1 == model2:
        st.warning("Please select two different models for comparison.")
    else:
        # Retrieve precision values
        precision1 = precisions[model1].get(disease, None)
        precision2 = precisions[model2].get(disease, None)

        if precision1 is not None and precision2 is not None:
            # Prepare data for bar chart
            labels = [model1, model2]
            precisions_values = [precision1, precision2]

            # Create smaller, interactive bar chart using Plotly
            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=precisions_values,
                    marker_color=['#636EFA', '#EF553B'],
                    text=[f"{v:.2f}" for v in precisions_values],
                    textposition='auto',
                    hoverinfo='y+text'
                )
            ])
            fig.update_layout(
                title=f"Precision Comparison for {disease}",
                xaxis_title="Models",
                yaxis_title="Precision",
                yaxis=dict(range=[0, 1]),
                template='plotly_dark',
                height=300,  # Reduced height for smaller graph
                margin=dict(l=40, r=40, t=60, b=40),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Numerical Precision Differences
            st.markdown("### 🧮 Precision Differences")
            difference = precision1 - precision2
            difference_text = f"""
            **{model1} Precision:** {precision1:.2f}  
            **{model2} Precision:** {precision2:.2f}  
            **Difference ({model1} - {model2}):** {difference:.2f}
            """
            st.write(difference_text)

            # Additional Interactive Visualizations

            # 1. Radar Chart
            st.markdown("---")
            st.subheader("🕸️ Radar Chart Comparison")
            categories = ["Brown Spot", "Healthy Leaf", "Rice Blast", "Bacterial Blight"]

            # Extract precision metrics for the selected models
            radar_data = pd.DataFrame({
                "Category": categories,
                model1: [precisions[model1].get(cat, 0) for cat in categories],
                model2: [precisions[model2].get(cat, 0) for cat in categories]
            })
            radar_data = radar_data.set_index('Category')

            fig_radar = go.Figure()
            for model in [model1, model2]:
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_data.loc[:, model],
                    theta=radar_data.index,
                    fill='toself',
                    name=model
                ))

            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Radar Chart of Precision Metrics",
                template='plotly_dark',
                height=400
            )

            st.plotly_chart(fig_radar, use_container_width=True)

            # 2. Heatmap of Precision Metrics
            st.markdown("---")
            st.subheader("🔥 Heatmap of Precision Metrics")
            # Combine precision data for heatmap
            heatmap_data = pd.DataFrame(precisions).T  # Models as rows, metrics as columns
            # Exclude 'All Classes' from heatmap
            heatmap_data = heatmap_data.drop(columns=["All Classes"])
            corr_matrix = heatmap_data.corr()

            fig_heatmap = px.imshow(corr_matrix,
                                    text_auto=True,
                                    aspect="auto",
                                    color_continuous_scale='Viridis',
                                    title='Correlation Heatmap of Precision Metrics',
                                    labels={'color': 'Correlation'})
            fig_heatmap.update_layout(template='plotly_dark', height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # 3. Parallel Coordinates Plot
            st.markdown("---")
            st.subheader("📏 Parallel Coordinates Plot")
            parallel_data = pd.DataFrame({
                "Model": [model1, model2],
                "Brown Spot Precision": precisions[model1]["Brown Spot"],
                "Healthy Leaf Precision": precisions[model1]["Healthy Leaf"],
                "Rice Blast Precision": precisions[model1]["Rice Blast"],
                "Bacterial Blight Precision": precisions[model1]["Bacterial Blight"],
                "All Classes mAP@0.5": [map_values[model1], map_values[model2]]
            })
            parallel_data = parallel_data.set_index('Model')

            fig_parallel = px.parallel_coordinates(parallel_data, 
                                                   dimensions=["Brown Spot Precision", "Healthy Leaf Precision", "Rice Blast Precision", "Bacterial Blight Precision", "All Classes mAP@0.5"],
                                                   color="All Classes mAP@0.5",
                                                   color_continuous_scale=px.colors.diverging.Tealrose,
                                                   color_continuous_midpoint=0.9,
                                                   labels={
                                                       "Brown Spot Precision": "Brown Spot",
                                                       "Healthy Leaf Precision": "Healthy Leaf",
                                                       "Rice Blast Precision": "Rice Blast",
                                                       "Bacterial Blight Precision": "Bacterial Blight",
                                                       "All Classes mAP@0.5": "mAP@0.5"
                                                   },
                                                   title="Comprehensive Model Performance Metrics")

            fig_parallel.update_layout(template='plotly_dark', height=500)
            st.plotly_chart(fig_parallel, use_container_width=True)

            # 4. Interactive Data Table with AgGrid
            st.markdown("---")
            st.subheader("📊 Interactive Precision Metrics Table")
            comparison_table = pd.DataFrame({
                "Metric": ["Brown Spot", "Healthy Leaf", "Rice Blast", "Bacterial Blight", "All Classes mAP@0.5"],
                model1: [
                    precisions[model1]["Brown Spot"],
                    precisions[model1]["Healthy Leaf"],
                    precisions[model1]["Rice Blast"],
                    precisions[model1]["Bacterial Blight"],
                    map_values[model1]
                ],
                model2: [
                    precisions[model2]["Brown Spot"],
                    precisions[model2]["Healthy Leaf"],
                    precisions[model2]["Rice Blast"],
                    precisions[model2]["Bacterial Blight"],
                    map_values[model2]
                ]
            })

            gb = GridOptionsBuilder.from_dataframe(comparison_table)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_default_column(enablePivot=True, enableValue=True, enableSorting=True, enableFilter=True)
            grid_options = gb.build()
            AgGrid(comparison_table, gridOptions=grid_options, enable_enterprise_modules=True, theme='streamlit')

            # 5. Download Button
            st.markdown("---")
            st.subheader("📥 Download Comparison Data")
            st.markdown(get_table_download_link(pd.DataFrame({
                "Model": [model1, model2],
                "Precision": [precision1, precision2]
            })), unsafe_allow_html=True)

            # 6. Additional Playful Tool: Fun Fact
            st.markdown("---")
            st.subheader("🎉 Fun Fact")
            fun_facts = [
                "Rice is a staple food for over half of the world's population!",
                "There are over 40,000 varieties of rice worldwide.",
                "Rice paddies can support a diverse range of aquatic life.",
                "The genetic diversity of rice is crucial for food security.",
                "YOLO stands for 'You Only Look Once', emphasizing its real-time detection capabilities."
            ]
            st.write(np.random.choice(fun_facts))

        else:
            st.error("Selected disease not found in precision data.")

elif page == "📂 Results by Model":
    st.title("📂 Results by Model")
    st.markdown("Select a model to view all detection results.")
    
    selected_model = st.selectbox("Select a model", ["YOLOv8.pt", "YOLOv10.pt", "YOLOv11.pt"])
    model_folder = {
        "YOLOv8.pt": yolov8_folder,
        "YOLOv10.pt": yolov10_folder,
        "YOLOv11.pt": yolov11_folder
    }[selected_model]
    result_images = [f for f in os.listdir(model_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if result_images:
        st.subheader(f"Results for {selected_model}")
        display_images_in_columns([os.path.join(model_folder, f) for f in result_images])
    else:
        st.write(f"🚫 No results found for {selected_model}.")

elif page == "📞 Contact Us":
    st.title("📞 Contact Us")
    st.markdown("For inquiries, please contact us at [support@riceleafdisease.com](mailto:support@riceleafdisease.com).")
    
    contact_form = """
    <form action="https://formsubmit.co/your-email@example.com" method="POST">
        <input type="text" name="name" placeholder="Your Name" required style="width: 100%; padding: 8px; margin-bottom: 10px;">
        <input type="email" name="email" placeholder="Your Email" required style="width: 100%; padding: 8px; margin-bottom: 10px;">
        <textarea name="message" placeholder="Your Message" required style="width: 100%; padding: 8px; margin-bottom: 10px;"></textarea>
        <button type="submit">Send Message</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)
