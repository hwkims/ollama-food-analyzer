import streamlit as st
import base64
import requests
import json
import io
import plotly.graph_objects as go
import traceback

# --- Configuration ---
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"  # Localhost Ollama address
OLLAMA_MODEL = "granite3.2-vision" # Make sure this model is pulled in Ollama
# OLLAMA_MODEL = "llava" # Alternative if you have llava
REQUEST_TIMEOUT = 120  # Increased timeout for potentially longer analysis
# Standard daily values (example based on 2000 kcal diet - VERY approximate for comparison)
# Source: FDA - These are just for rough visualization, NOT dietary advice.
STD_DAILY_CARBS_G = 275
STD_DAILY_PROTEIN_G = 50 # (Based on 10% kcal, varies greatly with individual)
STD_DAILY_FAT_G = 78 # (Based on 35% kcal)
# --- End Configuration ---

# --- Helper Functions ---

def safe_get(data, keys, default=None):
    """Safely get nested dictionary keys."""
    if not isinstance(keys, list):
        keys = [keys]
    temp = data
    try:
        for key in keys:
            if temp is None: return default
            # Handle potential list index access if needed within the path
            if isinstance(temp, list) and isinstance(key, int):
                 if 0 <= key < len(temp):
                      temp = temp[key]
                 else:
                      return default # Index out of bounds
            # Handle dictionary key access
            elif isinstance(temp, dict):
                 temp = temp.get(key) # Use .get for safety
            else:
                 return default # Cannot traverse further

        # Ensure numeric values are returned as numbers if possible
        if isinstance(temp, (int, float)):
            return temp
        # Try converting string numbers, handle potential errors
        try:
            return int(temp)
        except (ValueError, TypeError):
            try:
                return float(temp)
            except (ValueError, TypeError):
                return temp # Return as is if conversion fails
    except (KeyError, TypeError, IndexError):
        return default
    return temp if temp is not None else default


def create_macro_pie_chart(macros):
    """Creates a Plotly pie chart for macronutrient distribution."""
    labels = ['Carbohydrates', 'Protein', 'Fat']
    # Use safe_get to handle missing keys or non-numeric values gracefully
    values = [
        safe_get(macros, 'carbohydrates', 0),
        safe_get(macros, 'protein', 0),
        safe_get(macros, 'fat', 0)
    ]

    # Ensure values are numeric for the chart
    numeric_values = []
    for v in values:
        if isinstance(v, (int, float)):
            numeric_values.append(v)
        else:
            numeric_values.append(0) # Treat non-numeric as 0 for chart

    # Filter out zero values to avoid cluttering the chart
    valid_labels = [label for i, label in enumerate(labels) if numeric_values[i] > 0]
    valid_values = [value for value in numeric_values if value > 0]

    if not valid_values: # If all values are 0 or missing/non-numeric
        return None # Or return a message figure

    fig = go.Figure(data=[go.Pie(labels=valid_labels,
                                 values=valid_values,
                                 hole=.3, # Make it a donut chart
                                 pull=[0.05 if v > 0 else 0 for v in valid_values], # Slightly pull slices
                                 marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c'], # Blue, Orange, Green
                                 textinfo='percent+label', # Show percentage and label on slices
                                 insidetextorientation='radial'
                                )])
    fig.update_layout(
        title_text='Estimated Macronutrient Distribution (grams)',
        title_x=0.5, # Center title
        legend_title_text='Macronutrients',
        margin=dict(t=50, b=0, l=0, r=0), # Adjust margins
        # height=350 # Optional: set fixed height
        showlegend=False # Legend is redundant with text on slices
    )
    return fig

# --- Ollama API Call Function ---
def analyze_image_with_ollama(image_bytes):
    """
    Sends the image to the Ollama API for detailed analysis and returns structured results.
    """
    try:
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # --- Enhanced Ollama Prompt ---
        prompt = """Analyze the food item(s) shown in this image. Provide a detailed analysis strictly in JSON format. Include the following keys:
- "food_name": The common name of the dish or primary food item(s). Be specific if possible (e.g., "Pepperoni Pizza", "Chicken Caesar Salad").
- "cuisine_type": The likely cuisine style (e.g., "Italian", "Mexican", "Indian", "American", "Undetermined").
- "detailed_ingredients": A list of visible or highly likely core ingredients (e.g., ["Pasta", "Tomato Sauce", "Ground Beef", "Parmesan Cheese"]).
- "preparation_guess": A guess at the primary preparation method (e.g., "Fried", "Baked", "Grilled", "Steamed", "Raw", "Mixed").
- "estimated_serving_size_g": An estimated serving size in grams (provide a single number, e.g., 350). Acknowledge this is an estimate.
- "estimated_calories_kcal": Estimated total calories (kcal) for the serving size (provide a single number).
- "estimated_macronutrients_g": An object containing estimated macronutrient content in grams for the serving size:
    - "carbohydrates": Estimated grams (number).
    - "protein": Estimated grams (number).
    - "fat": Estimated grams (number).
- "potential_allergens": List common potential allergens visually identifiable or highly associated with the dish (e.g., ["Dairy", "Gluten"]). Preface list with a disclaimer like "Based on visual cues and typical recipes; cross-contamination risk not assessed." or state "None obvious".
- "confidence_level": A qualitative assessment of the estimation confidence ("High", "Medium", "Low") based on image clarity, typicality, and visibility of components.
- "notes": Any important notes, such as "Nutritional values are estimates based on visual interpretation and common recipes. Actual values vary significantly based on specific ingredients, preparation, and portion size. Allergen information is indicative, not exhaustive."

Output ONLY the JSON object. Do not include ANY introductory text, explanations, markdown formatting (like ```json), or anything else outside the JSON structure.

Example Output Format:
{
  "food_name": "Cheeseburger with Fries",
  "cuisine_type": "American",
  "detailed_ingredients": ["Beef Patty", "Cheese", "Bun", "Lettuce", "Tomato", "French Fries", "Ketchup (possible)"],
  "preparation_guess": "Grilled (Patty), Fried (Fries)",
  "estimated_serving_size_g": 550,
  "estimated_calories_kcal": 950,
  "estimated_macronutrients_g": {
    "carbohydrates": 80,
    "protein": 45,
    "fat": 50
  },
  "potential_allergens": ["Gluten (Bun)", "Dairy (Cheese)", "Sesame (possible on bun)"],
  "confidence_level": "Medium",
  "notes": "Nutritional values are estimates based on visual interpretation and common recipes. Actual values vary significantly based on specific ingredients, preparation, and portion size. Allergen information is indicative, not exhaustive."
}
"""
        # --- Ollama API Payload ---
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False,
            "options": {"temperature": 0.0}, # Low temp for more deterministic JSON
            "format": "json" # Explicitly request JSON output format
        }

        # --- Make the API Call ---
        # Using st.info or similar within this function can cause issues if called outside the main app flow
        # Print statements are safer for debugging background tasks/functions
        print(f"Sending request to Ollama model: {OLLAMA_MODEL}...")
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status() # Raise HTTPError for bad responses

        # --- Process Response ---
        response_data = response.json()
        ollama_output_str = response_data.get('response', '').strip()
        # print(f"DEBUG: Ollama raw output string:\n{ollama_output_str}") # For debugging

        if not ollama_output_str:
             return None, "Error: Received an empty response from the model.", ollama_output_str

        parsed_json = None
        error_message = None

        # Try parsing the response string directly as JSON
        try:
            # Sometimes the model might still wrap the JSON in markdown, try to remove it
            if ollama_output_str.startswith("```json"):
                ollama_output_str = ollama_output_str[7:]
            if ollama_output_str.endswith("```"):
                ollama_output_str = ollama_output_str[:-3]
            ollama_output_str = ollama_output_str.strip()

            # Handle potential leading/trailing non-JSON text (more robust)
            json_start = ollama_output_str.find('{')
            json_end = ollama_output_str.rfind('}')
            if json_start != -1 and json_end != -1 and json_start < json_end:
                potential_json_str = ollama_output_str[json_start:json_end+1]
            else:
                potential_json_str = ollama_output_str # Fallback if {} not found

            parsed_json = json.loads(potential_json_str)
            return parsed_json, None, ollama_output_str # Return parsed JSON, no error, raw string

        except json.JSONDecodeError as json_e:
            error_message = (f"Error: Failed to parse the model's response as JSON, even after cleanup attempts. "
                             f"The model might not have followed the format instructions. JSONDecodeError: {json_e}")
            # Don't use st.warning here, return the error message instead
            print(f"JSON Parsing Error: {error_message}") # Log error
            return None, error_message, ollama_output_str # No JSON, error, return raw output

    # --- Handle Request Exceptions ---
    except requests.exceptions.ConnectionError:
        err_msg = f"Connection Error: Could not connect to Ollama server at {OLLAMA_API_URL}. Ensure it's running and accessible."
        print(err_msg)
        return None, err_msg, None
    except requests.exceptions.Timeout:
        err_msg = f"Timeout Error: Request to Ollama server timed out after {REQUEST_TIMEOUT} seconds. The model might be taking too long or the server is overloaded."
        print(err_msg)
        return None, err_msg, None
    except requests.exceptions.RequestException as e:
         # Check for 404 which might mean model not found
        if e.response is not None and e.response.status_code == 404:
             model_error_msg = (f"Model Not Found Error: The model '{OLLAMA_MODEL}' might not be available on the Ollama server at {OLLAMA_API_URL}. "
                                f"Please ensure the model is pulled using `ollama pull {OLLAMA_MODEL}`.")
             print(model_error_msg)
             return None, model_error_msg, None
        else:
            err_msg = f"Request Error: An error occurred connecting to Ollama API: {e}"
            print(err_msg)
            return None, err_msg, None
    except Exception as e:
        err_msg = f"An unexpected error occurred during API call: {e}"
        print(err_msg)
        print("--- Unexpected Error Traceback ---")
        traceback.print_exc()
        print("---------------------------------")
        return None, err_msg, None


# --- Streamlit App UI ---

st.set_page_config(page_title="Enhanced Food Analyzer", layout="wide")
st.title("📸 Enhanced Food Image Analyzer 🍽️")
st.caption("Upload a food image for detailed analysis including estimated nutrients, ingredients, and more.")

# Initialize session state variables if they don't exist
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'raw_output' not in st.session_state:
    st.session_state.raw_output = None
if 'uploaded_file_id' not in st.session_state:
    st.session_state.uploaded_file_id = None
if 'image_bytes' not in st.session_state:
    st.session_state.image_bytes = None


# File uploader
uploaded_file = st.file_uploader(
    "Choose a food image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image file (JPG, PNG) for analysis."
)

# Variable to track if analysis should run
run_analysis = False

if uploaded_file is not None:
    # *** CORRECTED FILE ID GENERATION ***
    # Generate a unique ID for the current file upload using name and size
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    # Check if this is a new file based on our generated ID
    if current_file_id != st.session_state.get('uploaded_file_id'):
        # print("DEBUG: New file detected or first upload. Triggering analysis.") # Optional Debug
        run_analysis = True
        # Store the new file's ID and bytes in session state
        st.session_state.uploaded_file_id = current_file_id
        st.session_state.image_bytes = uploaded_file.getvalue()
        # Reset previous results for the new file
        st.session_state.analysis_result = None
        st.session_state.error_message = None
        st.session_state.raw_output = None
    # else: # If it's the same file ID, rely on existing session state data
    #     print("DEBUG: Same file ID detected, using cached results.") # Optional Debug

# Perform analysis ONLY if triggered by a new file upload
if run_analysis:
    with st.spinner(f"🧠 Analyzing image with {OLLAMA_MODEL}... This can take up to {REQUEST_TIMEOUT} seconds."):
        # Ensure image_bytes for analysis is definitely available
        if st.session_state.image_bytes:
            result_data, error_msg, raw_response = analyze_image_with_ollama(st.session_state.image_bytes)
            st.session_state.analysis_result = result_data
            st.session_state.error_message = error_msg
            st.session_state.raw_output = raw_response # Store raw output regardless of success
        else:
             # This case should ideally not happen if run_analysis is True due to above logic
             st.session_state.error_message = "Internal error: Analysis triggered but no image data found in session state."
             st.session_state.analysis_result = None
             st.session_state.raw_output = None
             print(st.session_state.error_message) # Log this internal error


# --- Display Area ---
# Only show columns if an image has been uploaded and its bytes are in session state
if st.session_state.image_bytes:
    col1, col2 = st.columns([0.4, 0.6]) # Adjust column width ratio if needed

    with col1:
        st.subheader("Uploaded Image:")
        try:
            st.image(st.session_state.image_bytes, caption="Image to be analyzed", use_column_width=True)
        except Exception as img_e:
            st.error(f"Could not display image: {img_e}")

    with col2:
        st.subheader("Analysis Results:")

        # Display error FIRST if it occurred during the analysis attempt
        if st.session_state.error_message:
            st.error(f"Analysis Failed: {st.session_state.error_message}", icon="🚨")
            if st.session_state.raw_output:
                st.subheader("Raw Model Output (if available):")
                # Use expander for potentially long raw output
                with st.expander("Click to view raw output"):
                    st.code(st.session_state.raw_output, language='text')
            # Even if there's an error, keep showing the image in col1

        # Display results only if analysis was successful (no error message)
        elif st.session_state.analysis_result:
            data = st.session_state.analysis_result
            st.success("Analysis complete!", icon="✅")

            # --- Use Tabs for organized results ---
            tab_titles = ["📊 Summary", "🍎 Nutrients", "📝 Details", "📄 Raw JSON"]
            tabs = st.tabs(tab_titles)

            with tabs[0]: # Summary
                st.markdown(f"**Food Name:** {safe_get(data, 'food_name', 'N/A')}")
                st.markdown(f"**Cuisine Type:** {safe_get(data, 'cuisine_type', 'N/A')}")
                st.markdown(f"**Estimated Serving Size:** `{safe_get(data, 'estimated_serving_size_g', 'N/A')} g`")
                st.markdown(f"**Confidence Level:** {safe_get(data, 'confidence_level', 'N/A')}")
                notes = safe_get(data, 'notes', 'No additional notes provided.')
                if notes:
                    st.info(f"**Notes:** {notes}", icon="ℹ️")

            with tabs[1]: # Nutrients
                calories = safe_get(data, 'estimated_calories_kcal', 'N/A')
                st.metric(label="Estimated Calories", value=f"{calories} kcal" if isinstance(calories, (int, float)) else 'N/A')

                macros = safe_get(data, 'estimated_macronutrients_g', {})
                if isinstance(macros, dict):
                    st.markdown("---")
                    st.markdown("**Estimated Macronutrients (per serving):**")
                    carb = safe_get(macros, 'carbohydrates', 0)
                    prot = safe_get(macros, 'protein', 0)
                    fat = safe_get(macros, 'fat', 0)

                    # Ensure they are numbers for display and calculation
                    carb_val = carb if isinstance(carb, (int, float)) else 0
                    prot_val = prot if isinstance(prot, (int, float)) else 0
                    fat_val = fat if isinstance(fat, (int, float)) else 0

                    st.markdown(f"- **Carbohydrates:** `{carb_val} g`")
                    st.markdown(f"- **Protein:** `{prot_val} g`")
                    st.markdown(f"- **Fat:** `{fat_val} g`")
                    st.markdown("---")

                    # --- Macronutrient Pie Chart ---
                    pie_fig = create_macro_pie_chart(macros) # Pass the original macros dict
                    if pie_fig:
                        st.plotly_chart(pie_fig, use_container_width=True)
                    else:
                        st.write("Macronutrient data not sufficient or invalid for chart.")

                    # --- Rough % Daily Value Comparison ---
                    st.markdown("---")
                    st.markdown("**Approximate % of Example Daily Values (2000 kcal diet)**")
                    st.caption("⚠️ **Disclaimer:** These percentages are based on *generic* 2000 kcal daily values (Carbs: 275g, Protein: 50g, Fat: 78g) for illustration ONLY. Your individual needs vary significantly. This is NOT dietary advice.")

                    carb_pct = (carb_val / STD_DAILY_CARBS_G) * 100 if STD_DAILY_CARBS_G > 0 else 0
                    prot_pct = (prot_val / STD_DAILY_PROTEIN_G) * 100 if STD_DAILY_PROTEIN_G > 0 else 0
                    fat_pct = (fat_val / STD_DAILY_FAT_G) * 100 if STD_DAILY_FAT_G > 0 else 0

                    # Use columns for better layout of progress bars
                    p_col1, p_col2, p_col3 = st.columns(3)
                    with p_col1:
                        st.text("Carbs:")
                        st.progress(min(int(carb_pct), 100), text=f"{carb_pct:.1f}%")
                    with p_col2:
                        st.text("Protein:")
                        st.progress(min(int(prot_pct), 100), text=f"{prot_pct:.1f}%")
                    with p_col3:
                         st.text("Fat:")
                         st.progress(min(int(fat_pct), 100), text=f"{fat_pct:.1f}%")

                else:
                    st.warning("Macronutrient data is missing or not in the expected dictionary format.")


            with tabs[2]: # Details
                st.markdown("**Detailed Ingredients (Estimated):**")
                ingredients = safe_get(data, 'detailed_ingredients', [])
                if isinstance(ingredients, list) and ingredients:
                    st.markdown("\n".join([f"- {item}" for item in ingredients]))
                else:
                    st.write("No specific ingredients listed or data unavailable.")

                st.markdown("---")
                st.markdown(f"**Guessed Preparation Method:** {safe_get(data, 'preparation_guess', 'N/A')}")
                st.markdown("---")

                st.markdown("**Potential Allergens (Visual/Typical Estimate):**")
                allergens = safe_get(data, 'potential_allergens', [])
                if isinstance(allergens, list) and allergens:
                    st.markdown("\n".join([f"- {item}" for item in allergens]))
                    st.warning("⚠️ Allergen list is based on visual cues & typical recipes. It's not exhaustive and doesn't account for cross-contamination. Always verify if you have allergies.", icon="❗")
                elif isinstance(allergens, str) and allergens: # Handle cases where it might be a string like "None obvious"
                     st.write(allergens)
                else:
                    st.write("No obvious potential allergens identified visually or data unavailable.")


            with tabs[3]: # Raw JSON
                # Display the structured JSON that was successfully parsed
                st.json(data)
                # Optionally show the original raw string from the model as well
                if st.session_state.raw_output:
                     with st.expander("View original raw string from model"):
                          st.code(st.session_state.raw_output, language='text')


        # Handle case where image is uploaded but analysis hasn't run or is pending (should be brief)
        elif not st.session_state.error_message and not st.session_state.analysis_result:
             st.info("Awaiting analysis results...") # Should usually be covered by spinner

# Show initial message if no file has ever been uploaded in the session
elif not st.session_state.uploaded_file_id:
    st.info("⬆️ Upload an image file above to start the analysis.")


# --- Footer ---
st.markdown("---")
st.caption(f"Powered by Ollama | Model: `{OLLAMA_MODEL}` | API: `{OLLAMA_API_URL}`")
st.caption("Disclaimer: All nutritional and ingredient information is estimated by an AI model based on the image and common data. Values can vary significantly. Not intended as dietary advice.")
