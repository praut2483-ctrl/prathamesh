import streamlit as st
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Doctor",
    page_icon="🌱",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🌱 Crop Doctor")
st.write(
    "AI-powered crop disease detection using Vision Transformer (ViT)"
)


# ============================================================
# MODEL PATHS
# ============================================================

model_paths = {
    "Corn": "./corn-vit-final",
    "Potato": "./potato-vit-final",
    "Rice": "./rice-vit-final",
    "Sugarcane": "./sugarcane-vit-final",
    "Wheat": "./wheat-vit-final"
}


# ============================================================
# DISEASE INFORMATION
# ============================================================

disease_info = {

    # --------------------------------------------------------
    # CORN
    # --------------------------------------------------------

    "Corn___Common_Rust": {
        "name": "Common Rust",
        "description": (
            "Common Rust is a fungal disease that affects corn leaves "
            "and can reduce photosynthesis and crop yield."
        ),
        "symptoms": (
            "Small reddish-brown to orange rust-colored spots appear "
            "on the upper and lower surfaces of corn leaves."
        ),
        "solution": (
            "Use resistant corn varieties. In severe cases, "
            "appropriate fungicides can be applied according to "
            "local agricultural recommendations."
        ),
        "prevention": (
            "Use resistant varieties, maintain good field management, "
            "and monitor plants regularly for early symptoms."
        )
    },

    "Corn___Gray_Leaf_Spot": {
        "name": "Gray Leaf Spot",
        "description": (
            "Gray Leaf Spot is a fungal disease that mainly affects "
            "corn leaves and can reduce photosynthesis and yield."
        ),
        "symptoms": (
            "Long, narrow gray or tan rectangular lesions develop "
            "on the leaves and may expand over time."
        ),
        "solution": (
            "Use resistant varieties and apply suitable fungicides "
            "when necessary. Follow recommended agricultural practices."
        ),
        "prevention": (
            "Remove or properly manage infected crop residue, "
            "rotate crops, and use resistant varieties."
        )
    },

    "Corn___Healthy": {
        "name": "Healthy Corn",
        "description": (
            "The corn leaf appears healthy and does not show clear "
            "visual symptoms of the diseases included in this model."
        ),
        "symptoms": (
            "Leaves generally have a healthy green appearance without "
            "significant disease lesions or abnormal discoloration."
        ),
        "solution": (
            "No disease treatment is required. Continue normal crop "
            "management, irrigation, and nutrient management."
        ),
        "prevention": (
            "Regularly inspect plants, maintain proper nutrition, "
            "control weeds, and monitor for early disease symptoms."
        )
    },

    "Corn___Northern_Leaf_Blight": {
        "name": "Northern Leaf Blight",
        "description": (
            "Northern Leaf Blight is a fungal disease that can cause "
            "significant damage to corn leaves."
        ),
        "symptoms": (
            "Large, elongated, gray-green or tan cigar-shaped lesions "
            "appear on the leaves."
        ),
        "solution": (
            "Use resistant varieties and consider appropriate fungicide "
            "applications when disease pressure is high."
        ),
        "prevention": (
            "Use resistant varieties, practice crop rotation, "
            "manage infected crop residue, and monitor fields regularly."
        )
    },


    # --------------------------------------------------------
    # POTATO
    # --------------------------------------------------------

    "Potato___Early_Blight": {
        "name": "Early Blight",
        "description": (
            "Early Blight is a fungal disease that commonly affects "
            "potato leaves and can reduce plant vigor and yield."
        ),
        "symptoms": (
            "Dark brown circular spots with concentric rings appear "
            "on older leaves. Yellowing may occur around the lesions."
        ),
        "solution": (
            "Remove severely infected plant material where practical "
            "and use appropriate fungicides according to local recommendations."
        ),
        "prevention": (
            "Use healthy seed potatoes, rotate crops, maintain good "
            "field sanitation, and avoid prolonged leaf wetness."
        )
    },

    "Potato___Healthy": {
        "name": "Healthy Potato",
        "description": (
            "The potato plant appears healthy without clear visual "
            "symptoms of Early Blight or Late Blight."
        ),
        "symptoms": (
            "Leaves appear green and healthy without significant "
            "disease spots or lesions."
        ),
        "solution": (
            "No disease treatment is required. Continue proper irrigation "
            "and nutrient management."
        ),
        "prevention": (
            "Use healthy planting material, inspect plants regularly, "
            "and maintain good field sanitation."
        )
    },

    "Potato___Late_Blight": {
        "name": "Late Blight",
        "description": (
            "Late Blight is a serious disease of potato that can spread "
            "rapidly under cool and wet conditions."
        ),
        "symptoms": (
            "Dark water-soaked lesions appear on leaves and may rapidly "
            "expand. White fungal growth may appear under humid conditions."
        ),
        "solution": (
            "Remove heavily infected plant material when appropriate "
            "and use recommended fungicides as part of an integrated "
            "disease management program."
        ),
        "prevention": (
            "Use certified disease-free seed, avoid excessive leaf wetness, "
            "monitor weather conditions, and remove infected crop debris."
        )
    },


    # --------------------------------------------------------
    # RICE
    # --------------------------------------------------------

    "Rice___Healthy": {
        "name": "Healthy Rice",
        "description": (
            "The rice plant appears healthy and does not show clear "
            "visual symptoms of the diseases included in this model."
        ),
        "symptoms": (
            "Leaves have a healthy green appearance without significant "
            "blast lesions."
        ),
        "solution": (
            "No disease treatment is required. Maintain proper water, "
            "nutrient, and crop management."
        ),
        "prevention": (
            "Use healthy seed, maintain balanced fertilization, "
            "monitor fields regularly, and manage weeds properly."
        )
    },

    "Rice___Leaf_Blast": {
        "name": "Rice Leaf Blast",
        "description": (
            "Rice Leaf Blast is a fungal disease that affects rice leaves "
            "and can reduce plant growth and yield."
        ),
        "symptoms": (
            "Spindle-shaped or diamond-shaped lesions with gray centers "
            "and darker borders may appear on the leaves."
        ),
        "solution": (
            "Use resistant varieties and apply suitable fungicides "
            "when recommended. Avoid excessive nitrogen application."
        ),
        "prevention": (
            "Use resistant varieties, maintain balanced nutrition, "
            "use healthy seed, and monitor fields regularly."
        )
    },

    "Rice___Neck_Blast": {
        "name": "Rice Neck Blast",
        "description": (
            "Rice Neck Blast affects the neck area of the rice panicle "
            "and can cause serious yield losses."
        ),
        "symptoms": (
            "Dark lesions develop around the neck of the panicle, "
            "which can cause panicles to become partially or completely "
            "unfilled."
        ),
        "solution": (
            "Use appropriate fungicide applications at recommended crop "
            "stages and follow local agricultural recommendations."
        ),
        "prevention": (
            "Use resistant varieties, maintain balanced nitrogen levels, "
            "and monitor the crop during panicle development."
        )
    },


    # --------------------------------------------------------
    # SUGARCANE
    # --------------------------------------------------------

    "Bacterial Blight": {
        "name": "Bacterial Blight",
        "description": (
            "Bacterial Blight is a bacterial disease that can affect "
            "sugarcane leaves and reduce plant health."
        ),
        "symptoms": (
            "Water-soaked or elongated lesions and discoloration may "
            "develop on leaves. Severe infection can cause leaf drying."
        ),
        "solution": (
            "Remove severely infected plants where practical and use "
            "healthy planting material. Follow local agricultural "
            "recommendations for disease management."
        ),
        "prevention": (
            "Use healthy disease-free planting material, maintain field "
            "sanitation, and monitor crops regularly."
        )
    },

    "Healthy": {
        "name": "Healthy Sugarcane",
        "description": (
            "The sugarcane plant appears healthy without clear visual "
            "symptoms of the diseases included in this model."
        ),
        "symptoms": (
            "Leaves appear green and healthy without significant disease "
            "lesions or abnormal discoloration."
        ),
        "solution": (
            "No disease treatment is required. Continue proper irrigation, "
            "fertilization, and crop management."
        ),
        "prevention": (
            "Use healthy planting material, maintain balanced nutrition, "
            "control weeds, and inspect plants regularly."
        )
    },

    "Red Rot": {
        "name": "Red Rot",
        "description": (
            "Red Rot is a serious fungal disease of sugarcane that can "
            "cause significant losses in crop production."
        ),
        "symptoms": (
            "The internal stalk tissue may develop a characteristic "
            "reddish discoloration with pale patches. Leaves may yellow "
            "and dry as the disease progresses."
        ),
        "solution": (
            "Remove and destroy infected stalks where appropriate and "
            "use disease-free planting material. Follow local agricultural "
            "recommendations for disease management."
        ),
        "prevention": (
            "Plant resistant varieties, use healthy seed cane, maintain "
            "field sanitation, and avoid planting infected material."
        )
    },


    # --------------------------------------------------------
    # WHEAT
    # --------------------------------------------------------

    "Wheat___Brown_Rust": {
        "name": "Brown Rust",
        "description": (
            "Brown Rust is a fungal disease of wheat that can reduce "
            "photosynthesis and grain production."
        ),
        "symptoms": (
            "Small orange-brown or reddish-brown pustules appear mainly "
            "on wheat leaves."
        ),
        "solution": (
            "Use resistant varieties and apply suitable fungicides "
            "when recommended based on disease severity."
        ),
        "prevention": (
            "Use resistant varieties, monitor crops regularly, "
            "and follow good crop management practices."
        )
    },

    "Wheat___Healthy": {
        "name": "Healthy Wheat",
        "description": (
            "The wheat plant appears healthy and does not show clear "
            "visual symptoms of Brown Rust or Yellow Rust."
        ),
        "symptoms": (
            "Leaves have a healthy green appearance without significant "
            "rust-colored pustules."
        ),
        "solution": (
            "No disease treatment is required. Continue proper irrigation "
            "and nutrient management."
        ),
        "prevention": (
            "Use healthy seed, select suitable varieties, monitor fields "
            "regularly, and maintain balanced crop nutrition."
        )
    },

    "Wheat___Yellow_Rust": {
        "name": "Yellow Rust",
        "description": (
            "Yellow Rust is a fungal disease that affects wheat leaves "
            "and can reduce crop productivity."
        ),
        "symptoms": (
            "Yellow or orange-yellow pustules often appear in stripes "
            "along the wheat leaves."
        ),
        "solution": (
            "Use resistant varieties and apply appropriate fungicides "
            "when recommended by local agricultural experts."
        ),
        "prevention": (
            "Use resistant varieties, monitor crops regularly, "
            "and manage the crop according to local recommendations."
        )
    }
}


# ============================================================
# CROP SELECTION
# ============================================================

crop = st.selectbox(
    "🌾 Select Crop",
    list(model_paths.keys())
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model(model_path):

    processor = ViTImageProcessor.from_pretrained(
        model_path
    )

    model = ViTForImageClassification.from_pretrained(
        model_path
    )

    model.eval()

    return processor, model


processor, model = load_model(
    model_paths[crop]
)


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📷 Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )


    if st.button("🔍 Predict Disease"):

        with st.spinner("Analyzing leaf..."):

            # --------------------------------------------
            # PROCESS IMAGE
            # --------------------------------------------

            inputs = processor(
                images=image,
                return_tensors="pt"
            )


            # --------------------------------------------
            # MODEL PREDICTION
            # --------------------------------------------

            with torch.no_grad():

                outputs = model(
                    **inputs
                )


            # --------------------------------------------
            # CALCULATE PROBABILITIES
            # --------------------------------------------

            probabilities = torch.nn.functional.softmax(
                outputs.logits,
                dim=-1
            )


            # --------------------------------------------
            # GET PREDICTION
            # --------------------------------------------

            confidence, predicted_class = torch.max(
                probabilities,
                dim=-1
            )


            # --------------------------------------------
            # GET DISEASE LABEL
            # --------------------------------------------

            predicted_label = model.config.id2label[
                predicted_class.item()
            ]


            # --------------------------------------------
            # CONFIDENCE %
            # --------------------------------------------

            confidence_value = (
                confidence.item() * 100
            )


        # ====================================================
        # DISPLAY PREDICTION
        # ====================================================

        st.success(
            "Prediction Complete! 🎉"
        )

        st.subheader(
            "🌿 Prediction Result"
        )

        st.write(
            f"**Crop:** {crop}"
        )

        st.write(
            f"**Disease:** {predicted_label}"
        )

        st.write(
            f"**Confidence:** {confidence_value:.2f}%"
        )


        # ====================================================
        # DISPLAY DISEASE INFORMATION
        # ====================================================

        if predicted_label in disease_info:

            info = disease_info[
                predicted_label
            ]


            st.divider()

            st.header(
                f"📖 About {info['name']}"
            )

            st.write(
                info["description"]
            )


            st.subheader(
                "🔍 Symptoms"
            )

            st.write(
                info["symptoms"]
            )


            st.subheader(
                "💊 Solution / Treatment"
            )

            st.write(
                info["solution"]
            )


            st.subheader(
                "🛡️ Prevention"
            )

            st.write(
                info["prevention"]
            )


        else:

            st.warning(
                "Information for this disease is not available yet."
            )