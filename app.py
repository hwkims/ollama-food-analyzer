import streamlit as st
import base64
import requests
import json
import io # Needed for handling image bytes with st.image

# --- Configuration ---
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # Localhost Ollama address
OLLAMA_MODEL = "granite3.2-vision" # Make sure this model is pulled in Ollama
# OLLAMA_MODEL = "llava" # Alternative if you have llava
REQUEST_TIMEOUT = 90  # Seconds timeout
# --- End Configuration ---

# --- Ollama API Call Function ---
def analyze_image_with_ollama(image_bytes):
    """
    Sends the image to the Ollama API for analysis and returns the result.
    """
    try:
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # --- Ollama Prompt (ensure it requests JSON) ---
        prompt = """Analyze the food item(s) shown in this image. Provide a detailed analysis strictly in JSON format. Include the following keys:
- "food_name": The common name of the dish or primary food item.
- "description": A brief description or list of main visible ingredients.
- "estimated_serving_size_g": An estimated serving size in grams (provide a single number, e.g., 250). Acknowledge this is an estimate.
- "estimated_calories_kcal": Estimated total calories (kcal) for the serving size (provide a single number).
- "estimated_macronutrients_g": An object containing estimated macronutrient content in grams for the serving size:
    - "carbohydrates": Estimated grams (number).
    - "protein": Estimated grams (number).
    - "fat": Estimated grams (number).
- "confidence_level": A qualitative assessment of the estimation confidence (e.g., "High", "Medium", "Low") based on image clarity and typicality of the dish.
- "notes": Any important notes, such as "Nutritional values are estimates based on visual interpretation and common recipes. Actual values can vary significantly depending on specific ingredients, preparation methods, and portion size."

Output ONLY the JSON object. Do not include any introductory text, explanations, markdown formatting, or anything else outside the JSON structure itself.

Example of the desired output format:
{
  "food_name": "Spaghetti Bolognese",
  "description": "Pasta with a meat-based tomato sauce, possibly topped with cheese.",
  "estimated_serving_size_g": 400,
  "estimated_calories_kcal": 650,
  "estimated_macronutrients_g": {
    "carbohydrates": 75,
    "protein": 30,
    "fat": 25
  },
  "confidence_level": "Medium",
  "notes": "Nutritional values are estimates based on visual interpretation and common recipes. Actual values can vary significantly depending on specific ingredients, preparation methods, and portion size."
}
"""
        # --- Ollama API Payload ---
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {"temperature": 0.0},
            "format": "json" # Request JSON format directly
        }

        # --- Make the API Call ---
        st.info(f"Sending request to Ollama model: {OLLAMA_MODEL}...")
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # --- Process Response ---
        response_data = response.json()
        ollama_output_str = response_data.get('response', '').strip()
        # print(f"Ollama raw output string:\n{ollama_output_str}") # For debugging in console

        if not ollama_output_str:
             return None, "Received an empty response from the model.", None # No JSON, error message, raw output

        parsed_json = None
        error_message = None
        result_json_str_formatted = None # For display if direct parsing works

        # Try parsing the response string directly as JSON
        try:
            parsed_json = json.loads(ollama_output_str)
            # Pretty-print the valid JSON
            result_json_str_formatted = json.dumps(parsed_json, indent=2, ensure_ascii=False)
            return parsed_json, None, None # Return parsed JSON, no error, no raw needed
        except json.JSONDecodeError:
            # If direct parsing fails, maybe the model added extra text
            # (although format: "json" should prevent this)
            # We'll return the raw string and an error message.
            error_message = ("The model's response was not valid JSON, despite requesting JSON format. "
                             "Displaying the raw response.")
            # Optionally, you could try extracting JSON from the string here as a fallback,
            # but format: "json" should ideally make this unnecessary.
            return None, error_message, ollama_output_str # No JSON, error, return raw output

    # --- Handle Request Exceptions ---
    except requests.exceptions.ConnectionError:
        return None, f"Connection Error: Could not connect to Ollama server at {OLLAMA_API_URL}. Please ensure it's running.", None
    except requests.exceptions.Timeout:
        return None, f"Timeout Error: Request to Ollama server timed out after {REQUEST_TIMEOUT} seconds.", None
    except requests.exceptions.RequestException as e:
        return None, f"Request Error: An error occurred during the request to Ollama API: {e}", None
    except Exception as e:
        import traceback
        print("--- Unexpected Error Traceback ---")
        traceback.print_exc()
        print("---------------------------------")
        return None, f"An unexpected error occurred: {e}", None


# --- Streamlit App UI ---

st.set_page_config(page_title="Food Image Analyzer", layout="wide")
st.title("🍕 Food Image Analyzer 🥗")
st.write("Upload an image of food, and the Ollama model will try to estimate its nutritional content.")

# File uploader
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image bytes
    image_bytes = uploaded_file.getvalue()

    # Create columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image:")
        # Display the uploaded image using bytes
        st.image(image_bytes, caption=uploaded_file.name, use_column_width=True)

    with col2:
        st.subheader("Analysis Results:")
        # Analyze the image when the file is uploaded
        with st.spinner(f"Analyzing image with {OLLAMA_MODEL}... This may take a minute."):
            result_data, error_msg, raw_output = analyze_image_with_ollama(image_bytes)

        if error_msg:
            st.error(error_msg)
            if raw_output:
                st.text("Raw response from model:")
                st.code(raw_output, language='text')
        elif result_data:
            st.success("Analysis complete!")
            # Display the structured JSON result
            st.json(result_data)
        else:
            # Should not happen if analyze_image_with_ollama is correct, but as a fallback
            st.error("An unknown issue occurred during analysis.")

else:
    st.info("Please upload an image file to start the analysis.")

st.markdown("---")
st.caption(f"Using Ollama model: `{OLLAMA_MODEL}` via `{OLLAMA_API_URL}`")
