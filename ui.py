# --- Import necessary libraries ---
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pdfminer.high_level import extract_text
import docx2txt
from PIL import Image
from typing import List, Dict, Any, Optional
import base64
import re
import io
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import uvicorn
import mimetypes # To help determine file types

# --- Load Environment Variables ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
# Allows requests from any origin (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---

def extract_txt(file_content: bytes, filename: str) -> str:
    """Extracts text from plain text file content."""
    try:
        # Try decoding with utf-8 first, fallback to latin-1 if needed
        try:
            text = file_content.decode("utf-8")
        except UnicodeDecodeError:
            text = file_content.decode("latin-1") # Common fallback
        return text.strip()
    except Exception as e:
        print(f"Error reading text file {filename}: {e}")
        return f"Error reading file '{filename}': {e}"

def extract_pdf(file_content: bytes, filename: str) -> str:
    """Extracts text from PDF file content."""
    try:
        pdf_stream = io.BytesIO(file_content)
        text = extract_text(pdf_stream)
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        return text
    except Exception as e:
        print(f"Error reading PDF file {filename}: {e}")
        return f"Error reading file '{filename}': {e}"

def extract_docx(file_content: bytes, filename: str) -> str:
    """Extracts text from DOCX file content."""
    try:
        docx_stream = io.BytesIO(file_content)
        text = docx2txt.process(docx_stream)
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        return text
    except Exception as e:
        print(f"Error reading DOCX file {filename}: {e}")
        return f"Error reading file '{filename}': {e}"

def extract_img_data_uri(file_content: bytes, filename: str) -> Optional[str]:
    """Encodes image file content to a base64 data URI."""
    try:
        # Use Pillow to determine the image format safely
        with Image.open(io.BytesIO(file_content)) as img:
            img_format = img.format or 'jpeg' # Default to jpeg if format is unknown
        content_type = f"image/{img_format.lower()}"
        base64_image = base64.b64encode(file_content).decode("utf-8")
        return f"data:{content_type};base64,{base64_image}"
    except Exception as e:
        print(f"Error processing image file {filename}: {e}")
        return None # Indicate failure

def generate_prompt(developer_instructions: str, extracted_texts: Dict[str, str], image_filenames: List[str]) -> str:
    """
    Generates the prompt for the LLM based on instructions and processed files,
    aiming to create various software development artifacts.
    """

    context_str = "**BEGIN TASK CONTEXT**\n\n"

    # --- Add Text Documents Context ---
    if extracted_texts:
        context_str += "### Provided Text Documents:\n\n"
        for filename, text in extracted_texts.items():
            context_str += f"--- Start of content from: {filename} ---\n"
            # Include the full text content without truncation
            context_str += f"{text}\n"
            context_str += f"--- End of content from: {filename} ---\n\n"
    else:
        context_str += "### Provided Text Documents:\nNo text documents were provided or successfully processed.\n\n"

    # --- Add Image Files Context ---
    if image_filenames:
        context_str += "### Provided Image Files:\n"
        for filename in image_filenames:
            context_str += f"- {filename}\n"
        context_str += "(Note: The visual content of these images is provided separately to the model if they were processed successfully.)\n\n"
    else:
        context_str += "### Provided Image Files:\nNo image files were provided.\n\n"

    context_str += "**END TASK CONTEXT**\n\n"

    # --- Construct the Final Prompt ---
    prompt = f"""**Role:** You are an AI assistant specialized in software development tasks, particularly **frontend/UI development**. Your capabilities include analyzing requirements, interpreting UI designs/wireframes, and generating functional code using standard web technologies (HTML, CSS, JavaScript) and popular frameworks (React, Vue, Angular, etc.).

**Objective:** Based *strictly* on the provided context (Text Documents, Image Files, Wireframe Descriptions listed above, if any) and the specific 'Instructions' below, generate **functional frontend/UI code** for the requested interface component or screen. Synthesize information accurately from all provided sources to implement the specified structure, elements, basic styling, and interactions.

{context_str}
### Instructions (Your Task)
{developer_instructions}
*Focus on generating clean, well-structured, and potentially reusable frontend code (e.g., HTML structure, CSS for styling, JavaScript for basic interactions, or code within a specified framework like React, Vue, Angular). Implement the visual layout, UI elements (buttons, forms, navigation, etc.), basic styling (layout, colors, fonts if specified), and essential user interactions described or implied in the context/instructions. Adhere to common frontend development best practices.*

### Output Requirements
- Generate *only* the specific **frontend/UI code** requested in the 'Instructions'. If multiple technologies are required (e.g., HTML, CSS, JS), provide code blocks for each.
- Ensure the generated code directly addresses the instructions and accurately implements the structure, elements, styling hints, and interactions described in the provided context.
- Format the output using appropriate code blocks, clearly indicating the language (e.g., HTML, CSS, JavaScript, JSX, Vue SFC) or file name if applicable. Separate code for different languages or components logically.
- Do not add conversational introductions, conclusions, explanations about the code, or import statements unless they are essential for the core functionality requested or explicitly asked for. Focus solely on generating the requested code snippet(s)."""
    return prompt.strip()

def generate_llm_response(prompt: str, image_data_uris: List[Dict[str, str]]) -> str:
    """
    Generates response from the LLM, handling optional image inputs.
    image_data_uris is a list of dicts: [{'filename': str, 'uri': str}]
    """
    if not google_api_key:
        raise HTTPException(status_code=500, detail="Server configuration error: Google API Key not set.")

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Use a model supporting vision like flash or pro-vision
            temperature=0.35,
            max_tokens=30000, # Consider adjusting based on expected output size
            google_api_key=google_api_key
        )

        # --- Construct the message content ---
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        # Add image data if available
        for img_data in image_data_uris:
            content.append({
                "type": "image_url",
                "image_url": {"url": img_data['uri']}
            })
            print(f"Added image {img_data['filename']} to LLM input.") # Log image addition

        # Create the message for the LLM
        message = HumanMessage(content=content)

        # Invoke the LLM
        print("Invoking LLM...") # Log before invocation
        response = llm.invoke([message])
        print("LLM invocation complete.") # Log after invocation

        return response.content

    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        # Re-raise as HTTPException to send proper error response to client
        raise HTTPException(status_code=500, detail=f"Error generating response from AI model: {e}")


# --- FastAPI Endpoint ---

@app.post("/ui")
async def process_instruction(
    # --- Mandatory Parameter ---
    developer_instructions: str = Form(...),
    # --- Optional List of Files ---
    # Default to an empty list if no files are provided
    files: List[UploadFile] = File([])
):
    """
    Processes developer instructions with an optional list of context files (text, pdf, docx, images).

    - **developer_instructions**: The mandatory instructions for the developer (string).
    - **files**: An optional list of files to provide context. Supports .txt, .pdf, .docx, .png, .jpg, .jpeg, .webp, .gif.
    """
    extracted_texts: Dict[str, str] = {}
    image_data_uris: List[Dict[str, str]] = [] # Store as {'filename': ..., 'uri': ...}
    processed_filenames: List[str] = [] # Keep track of files processed

    print(f"Received request with {len(files)} file(s).")

    try:
        # --- Process Uploaded Files ---
        for file in files:
            filename = file.filename or "unknown_file"
            processed_filenames.append(filename)
            print(f"Processing file: {filename}, Content-Type: {file.content_type}")

            # Read file content once
            file_content = await file.read()
            await file.seek(0) # Reset pointer just in case it's needed later (though read() consumes it)

            # Determine file type and process
            content_type = file.content_type
            file_ext = Path(filename).suffix.lower()

            # --- Image Processing ---
            if content_type and content_type.startswith('image/'):
                print(f"Attempting to process {filename} as image.")
                uri = extract_img_data_uri(file_content, filename)
                if uri:
                    image_data_uris.append({"filename": filename, "uri": uri})
                    print(f"Successfully processed {filename} as image.")
                else:
                    # Store error message if image processing failed
                    extracted_texts[filename] = f"Error processing image file '{filename}'."
                    print(f"Failed to process {filename} as image.")

            # --- Text/Document Processing ---
            elif file_ext == '.pdf' or content_type == 'application/pdf':
                print(f"Attempting to process {filename} as PDF.")
                extracted_texts[filename] = extract_pdf(file_content, filename)
            elif file_ext == '.docx' or content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                 print(f"Attempting to process {filename} as DOCX.")
                 extracted_texts[filename] = extract_docx(file_content, filename)
            elif file_ext == '.txt' or content_type == 'text/plain':
                 print(f"Attempting to process {filename} as TXT.")
                 extracted_texts[filename] = extract_txt(file_content, filename)

            # --- Unsupported File Type ---
            else:
                print(f"Unsupported file type for {filename}: {content_type} / {file_ext}")
                extracted_texts[filename] = f"Unsupported file type ('{content_type}' / '{file_ext}'). Cannot process content."

            # Close the file explicitly after reading (important for temp files)
            # await file.close() # UploadFile is automatically closed by FastAPI after the request

        # --- Generate Prompt ---
        print("Generating prompt...")
        image_filenames = [img['filename'] for img in image_data_uris]
        prompt = generate_prompt(
            developer_instructions,
            extracted_texts,
            image_filenames
        )
        # print(f"Generated Prompt:\n{prompt[:500]}...") # Log prompt start for debugging

        # --- Generate LLM Response ---
        print("Generating LLM response...")
        response_content = generate_llm_response(prompt, image_data_uris)

        return {"response": response_content}

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions directly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected error occurred in /instruct endpoint: {e}")
        import traceback
        traceback.print_exc() # Log the full traceback for debugging
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")
    finally:
        # Ensure files are closed (FastAPI usually handles this, but good practice)
        for file in files:
             # Check if file object has close method and is not already closed
             if hasattr(file, 'close') and callable(file.close) and not getattr(file.file, 'closed', True):
                 try:
                     await file.close()
                     print(f"Closed file: {file.filename}")
                 except Exception as close_err:
                     print(f"Error closing file {file.filename}: {close_err}")


# --- Run the App (for local development) ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)