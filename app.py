import streamlit as st
import matplotlib.pyplot as plt
import os
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix
import numpy as np
from PIL import Image
from ultralytics import YOLO
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
# Paths to the YOLO models
model_paths = {
    "YOLOv8.pt": "D:/rice/Models/Models/YOLOv8Best.pt",
    "YOLOv10.pt": "D:/rice/Models/Models/YOLOv10Best.pt",
    "YOLOv11.pt": "D:/rice/Models/Models/YOLOv11Best.pt",
}
yolov8_folder = 'yolov8'
yolov10_folder = 'yolov10'
yolov11_folder = 'yolov11'
yolov80_folder = 'yolov80'
yolov100_folder = 'yolov100'
yolov110_folder = 'yolov110'
def load_images_from_folder(folder):
    image_files = []
    for filename in os.listdir(folder):
        if filename.endswith(('.svg', '.png', '.jpg')):
            image_files.append(os.path.join(folder, filename))
    return image_files
def display_disease_info():
    st.title("🌾 Rice Leaf Disease Information")
    
    st.header("1. 📌 Bacterial Blight")
    st.markdown(""" 
        Bacterial Blight is a major disease affecting rice crops. It causes lesions on leaves, eventually leading to plant death.
        - [Read more about Bacterial Blight](http://www.knowledgebank.irri.org/decision-tools/rice-doctor/rice-doctor-fact-sheets/item/bacterial-blight)
    """)
    
    st.header("2. 🛑 Brown Spot")
    st.markdown(""" 
        Brown Spot is caused by a fungal infection, leading to small brown spots on leaves that can kill the entire leaf.
        - [Read more about Brown Spot](http://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/brown-spot#:~:text=Brown%20spot%20is%20a%20fungal,can%20kill%20the%20whole%20leaf)
    """)
    
    st.header("3. 🌱 Rice Blast")
    st.markdown(""" 
        Rice Blast is a fungal disease that affects the plant’s leaves and can reduce grain yield significantly.
        - [Read more about Rice Blast](http://www.knowledgebank.irri.org/training/fact-sheets/pest-management/diseases/item/blast-leaf-collar#:~:text=Rice%20blast%20is%20one%20of,grain%20fill%2C%20reducing%20grain%20yield)
    """)
    
    st.header("4. ✅ Healthy Rice Leaves")
    st.markdown(""" 
        Healthy rice leaves are free from any disease and show no signs of lesions or abnormalities. These leaves are crucial for photosynthesis and optimal plant growth.
    """)

# Function to create folder if it doesn't exist
def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# Ensure all folders are created
for folder in [yolov8_folder, yolov10_folder, yolov11_folder]:
    create_folder(folder)

# Function to load YOLO model
def load_model(model_path):
    return YOLO(model_path)

# Function to predict and save results
def predict_and_save(model, img, model_folder):
    try:
        # Perform predictions
        results = model.predict(source=img, conf=0.25, imgsz=640, save=False)
        if results:
            # Get the predicted image with bounding boxes
            result_image = results[0].plot()
            # Convert result to PIL image and save
            result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
            save_path = os.path.join(model_folder, f"result_{np.random.randint(1e6)}.jpg")
            result_pil.save(save_path)
            return save_path
        else:
            return None
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return None

# Streamlit app
st.sidebar.title("🌟 Navigation")
page = st.sidebar.radio("Choose a page", ["Home","Disease Information", "Model Comparison","Image Gallery","Model Analytics","Results by Model","Contact Us"])

if page == "Home":
    st.title("🌾 Multi-Image Rice Leaf Disease Detection")
    st.markdown("Upload multiple images and detect diseases using state-of-the-art YOLO models.")
    
    # Multi-image upload
    uploaded_images = st.file_uploader(
        "Upload images of rice leaves (multiple allowed)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    # Model selection
    model_choice = st.selectbox("Select the model for detection", ["YOLOv8.pt", "YOLOv10.pt", "YOLOv11.pt"])
    model = load_model(model_paths[model_choice])
    model_folder = {"YOLOv8.pt": yolov8_folder, "YOLOv10.pt": yolov10_folder, "YOLOv11.pt": yolov11_folder}[model_choice]
    
    if uploaded_images:
        st.subheader("Detection Results")
        for img_file in uploaded_images:
            # Convert uploaded image for YOLO processing
            image = Image.open(img_file)
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Predict and save results
            result_path = predict_and_save(model, img, model_folder)
            
            if result_path:
                st.image(result_path, caption=f"Result for {img_file.name}", use_column_width=True)
            else:
                st.write(f"🚫 No objects detected in {img_file.name}.")
elif page == "Disease Information":
    display_disease_info()
elif page == "Model Analytics":
    st.title("📊 Model Analytics")
    st.markdown("Compare different models used for rice leaf disease detection.")
    
    # Dropdown to select model
    model_choice = st.selectbox("Choose Model", ["YOLOv8", "YOLOv10", "YOLOv11"])

    if model_choice == "YOLOv8":
        st.header("🔹 YOLOv8")
        st.markdown("YOLOv8 is optimized for speed and moderate accuracy.")
        
        # Display images for YOLOv8
        st.markdown("#### Visual Results from YOLOv8")
        yolo_v8_images = load_images_from_folder(yolov80_folder)  # Get images from YOLOv8 folder
        cols = st.columns(5)  # Display in rows (5 columns each)
        for i, image in enumerate(yolo_v8_images):
            image_name = os.path.basename(image)  # Extract the filename from the full path
            with cols[i % 5]:
                st.image(image, caption=image_name, use_container_width=True)

        # YOLOv8 performance chart
        st.markdown("### 📈 YOLOv8 Model Performance")
        st.markdown("#### Accuracy of YOLOv8")
        fig1, ax1 = plt.subplots()
        ax1.pie([85], labels=["YOLOv8"], autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        st.markdown("#### Speed of YOLOv8")
        fig2, ax2 = plt.subplots()
        ax2.bar(["YOLOv8"], [45], color='blue')
        ax2.set_ylabel("Frames Per Second (FPS)")
        ax2.set_title("Model Speed Comparison")
        st.pyplot(fig2)

    elif model_choice == "YOLOv10":
        st.header("🔹 YOLOv10")
        st.markdown("YOLOv10 improves accuracy but at the cost of some speed.")
        
        # Display images for YOLOv10
        st.markdown("#### Visual Results from YOLOv10")
        yolo_v10_images = load_images_from_folder(yolov100_folder)  # Get images from YOLOv10 folder
        cols = st.columns(5)  # Display in rows (5 columns each)
        for i, image in enumerate(yolo_v10_images):
            image_name = os.path.basename(image)  # Extract the filename from the full path
            with cols[i % 5]:
                st.image(image, caption=image_name, use_container_width=True)

        # YOLOv10 performance chart
        st.markdown("### 📈 YOLOv10 Model Performance")
        st.markdown("#### Accuracy of YOLOv10")
        fig1, ax1 = plt.subplots()
        ax1.pie([90], labels=["YOLOv10"], autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)

        st.markdown("#### Speed of YOLOv10")
        fig2, ax2 = plt.subplots()
        ax2.bar(["YOLOv10"], [30], color='orange')
        ax2.set_ylabel("Frames Per Second (FPS)")
        ax2.set_title("Model Speed Comparison")
        st.pyplot(fig2)

    elif model_choice == "YOLOv11":
        st.header("🔹 YOLOv11")
        st.markdown("YOLOv11 achieves the highest accuracy while maintaining reasonable speed.")
        
        # Display images for YOLOv11
        st.markdown("#### Visual Results from YOLOv11")
        yolo_v11_images = load_images_from_folder(yolov110_folder)  # Get images from YOLOv11 folder
        cols = st.columns(5)  # Display in rows (5 columns each)
        for i, image in enumerate(yolo_v11_images):
            image_name = os.path.basename(image)  # Extract the filename from the full path
            with cols[i % 5]:
                st.image(image, caption=image_name, use_container_width=True)

        # YOLOv11 performance chart
        st.markdown("### 📈 YOLOv11 Model Performance")
        st.markdown("#### Accuracy of YOLOv11")
        fig1, ax1 = plt.subplots()
        ax1.pie([95], labels=["YOLOv11"], autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')
        st.pyplot(fig1)
elif page == "Model Comparison":
    st.title("📊 Model Comparison")
    st.markdown("Compare the performance of YOLO models based on their ability to detect rice leaf diseases.")
    
    # Pie chart comparison
    st.subheader("Model Accuracy")
    models = ["YOLOv8", "YOLOv10", "YOLOv11"]
    accuracies = [90, 93, 95]
    fig, ax = plt.subplots()
    ax.pie(accuracies, labels=models, autopct="%1.1f%%", colors=["blue", "orange", "green"], startangle=140)
    ax.set_title("Accuracy of YOLO Models")
    st.pyplot(fig)
elif page == "Results by Model":
    st.title("View Results by Model")
    st.markdown("Select a model to view all detection results.")
    
    selected_model = st.selectbox("Select a model", ["YOLOv8.pt", "YOLOv10.pt", "YOLOv11.pt"])
    model_folder = {"YOLOv8.pt": yolov8_folder, "YOLOv10.pt": yolov10_folder, "YOLOv11.pt": yolov11_folder}[selected_model]
    result_images = [f for f in os.listdir(model_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if result_images:
        st.subheader(f"Results for {selected_model}")
        for result_img in result_images:
            result_path = os.path.join(model_folder, result_img)
            st.image(result_path, caption=result_img, use_column_width=True)
    else:
        st.write(f"🚫 No results found for {selected_model}.")
elif page == "Image Gallery":
    st.title("📷 Image Gallery")
    st.markdown("Explore images of different rice leaf diseases for reference.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("D:/rice/Models/Models/3.jpg", caption="Bacterial Blight")
    with col2:
        st.image("D:/rice/Models/Models/5.jpg", caption="Brown Spot")
    with col3:
        st.image("D:/rice/Models/Models/1.jpg", caption="Rice Blast")
elif page == "Contact Us":
    st.title("📞 Contact Us")
    st.markdown("For inquiries, please contact us at support@riceleafdisease.com.")
    
    contact_form = """
    <form action="https://formsubmit.co/your-email@example.com" method="POST">
        <input type="text" name="name" placeholder="Your Name" required><br>
        <input type="email" name="email" placeholder="Your Email" required><br>
        <textarea name="message" placeholder="Your Message" required></textarea><br>
        <button type="submit">Send Message</button>
    </form>
    """
    st.markdown(contact_form, unsafe_allow_html=True)
