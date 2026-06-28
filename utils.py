import os
import json
import shutil
import cv2
import pytesseract

from PIL import Image
from groq import Groq

# =========================================================
# GROQ CLIENT INITIALIZATION
# =========================================================

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it in your .env file.")

client = Groq(
    api_key=groq_api_key
)

print("[OK] Groq Client initialized successfully")

# =========================================================
# TESSERACT CONFIGURATION
# =========================================================

# Optional custom path
# Change path if needed

tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

# =========================================================
# ALLOWED FILE TYPES
# =========================================================

ALLOWED_EXTENSIONS = {
    'png',
    'jpg',
    'jpeg',
    'gif',
    'webp'
}

# =========================================================
# CHECK FILE TYPE
# =========================================================

def allowed_file(filename):

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def preprocess_image(image_path):

    """
    Preprocess image for better OCR
    """

    try:

        img = cv2.imread(image_path)

        gray = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

        # Thresholding
        _, thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Noise removal
        denoised = cv2.fastNlMeansDenoising(
            thresh,
            None,
            10,
            7,
            21
        )

        # Contrast enhancement
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(denoised)

        return enhanced

    except Exception as e:

        print(f"Preprocessing Error: {e}")

        return cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

# =========================================================
# OCR TEXT EXTRACTION
# =========================================================

def extract_text_from_image(image_path):

    """
    Extract text from image using Tesseract OCR
    """

    try:

        preprocessed = preprocess_image(image_path)

        pil_image = Image.fromarray(preprocessed)

        # OCR Config
        custom_config = r'--oem 3 --psm 6'

        extracted_text = pytesseract.image_to_string(
            pil_image,
            config=custom_config
        )

        # Retry if empty
        if not extracted_text.strip():

            custom_config = r'--oem 3 --psm 11'

            extracted_text = pytesseract.image_to_string(
                pil_image,
                config=custom_config
            )

        return extracted_text

    except Exception as e:

        return f"OCR Error: {str(e)}"

# =========================================================
# AI ANALYSIS FUNCTION
# =========================================================

def get_full_analysis_from_text(text, user_profile=None):

    """
    Analyze ingredients using Groq AI
    """

    profile_text = ""

    if isinstance(user_profile, dict):

        for key, value in user_profile.items():

            profile_text += f"{key}: {value}\n"

    prompt = f"""
You are an expert nutrition and food safety analyst.

PRODUCT LABEL TEXT:
{text}

USER PROFILE:
{profile_text}

Perform:

1. Extract:
   - Product name
   - Ingredients
   - Product type

2. Analyze:
   - Safety score
   - Harmful ingredients
   - Recommendations
   - Precautionary tips

Return ONLY valid JSON in this format:

{{
    "extraction": {{
        "product_name": "",
        "ingredients": [],
        "product_type": ""
    }},
    "analysis": {{
        "overall_safety_score": 0,
        "traffic_light": "",
        "summary": "",
        "harmful_ingredients": [
            {{
                "ingredient": "",
                "reason": "",
                "description": "",
                "identification": ""
            }}
        ],
        "recommendations": [],
        "precautionary_tips": []
    }}
}}

Rules:
- Safety score = 0-100
- Green >=80
- Yellow 50-79
- Red <50
- Return ONLY JSON
"""

    try:

        response = client.chat.completions.create(

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            model="llama-3.3-70b-versatile",

            temperature=0.3,

            max_tokens=2048
        )

        return response.choices[0].message.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        return json.dumps({
            "error": str(e)
        })

# =========================================================
# JSON PARSER
# =========================================================

def parse_json_response(response_text):

    """
    Parse AI JSON safely
    """

    try:

        json_str = response_text.strip()

        json_str = json_str.replace(
            '```json',
            ''
        ).replace(
            '```',
            ''
        )

        return json.loads(json_str)

    except json.JSONDecodeError:

        print("JSON Decode Error")
        print(response_text)

        return {
            "error": "Failed to parse AI response"
        }

    except Exception as e:

        return {
            "error": str(e)
        }

# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # IMAGE PATH
    # -----------------------------------------------------

    image_path = "sample.jpg"

    # -----------------------------------------------------
    # OCR EXTRACTION
    # -----------------------------------------------------

    extracted_text = extract_text_from_image(
        image_path
    )

    print("\n========== EXTRACTED TEXT ==========\n")

    print(extracted_text)

    # -----------------------------------------------------
    # USER PROFILE
    # -----------------------------------------------------

    user_profile = {
        "name": "John",
        "age": 25,
        "gender": "Male",
        "dietary_preferences": "Vegetarian",
        "allergies": "Milk",
        "medical_conditions": "Diabetes",
        "lifestyle_habits": "Gym"
    }

    # -----------------------------------------------------
    # AI ANALYSIS
    # -----------------------------------------------------

    ai_response = get_full_analysis_from_text(
        extracted_text,
        user_profile
    )

    print("\n========== RAW AI RESPONSE ==========\n")

    print(ai_response)

    # -----------------------------------------------------
    # JSON PARSE
    # -----------------------------------------------------

    parsed_response = parse_json_response(
        ai_response
    )

    print("\n========== FINAL JSON ==========\n")

    print(
        json.dumps(
            parsed_response,
            indent=4
        )
    )
