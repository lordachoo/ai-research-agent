#!/bin/bash

# Script to add documents (files or directories) to the AI Research Agent

# --- Configuration ---
PYTHON_MAIN_SCRIPT="app.main" # Relative to the project root
PROJECT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # Assumes script is in project root
VENV_PYTHON_PATH="${PROJECT_ROOT_DIR}/venv/bin/python"
SUPPORTED_EXTENSIONS=(".pdf" ".docx" ".md" ".markdown" ".csv" ".txt")
# --- End Configuration ---

# Function to display usage
usage() {
    echo "Usage: $0 <path_to_file_or_directory>"
    echo "Adds the specified file or all supported documents in the specified directory (recursively) to the AI Research Agent's knowledge base."
    echo "Supported file types: ${SUPPORTED_EXTENSIONS[*]}"
    echo ""
    echo "IMPORTANT: Ensure your Python virtual environment ('venv') is activated before running this script."
    echo "           (e.g., 'source venv/bin/activate' from the project root)"
    echo "           Then run this script: './add_docs.sh ./my_research_papers/'"
    exit 1
}

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Error: No path provided."
    usage
fi

INPUT_PATH="$1"
PYTHON_EXECUTABLE="python" # Default to 'python' assuming venv is active

# Check if virtual environment is active (basic check by looking for VIRTUAL_ENV)
# If not, try to use the specific venv python path.
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Python virtual environment does not appear to be active."
    echo "Please activate it first for best results: source \"${PROJECT_ROOT_DIR}/venv/bin/activate\""
    echo "Attempting to use Python directly from venv: ${VENV_PYTHON_PATH}"
    if [ -x "${VENV_PYTHON_PATH}" ]; then
        PYTHON_EXECUTABLE="${VENV_PYTHON_PATH}"
    else
        echo "Error: Python executable not found at ${VENV_PYTHON_PATH}."
        echo "Please ensure your virtual environment is set up correctly at '${PROJECT_ROOT_DIR}/venv' and/or activate it."
        exit 1
    fi
fi

# Function to process a single file
process_file() {
    local file_path="$1"
    echo "--------------------------------------------------"
    echo "Attempting to add file: $file_path"
    # Run the python command from the project root directory
    (cd "$PROJECT_ROOT_DIR" && "$PYTHON_EXECUTABLE" -m "$PYTHON_MAIN_SCRIPT" add-document "$file_path")
    echo "--------------------------------------------------"
}

# Check if the input path exists
if [ ! -e "$INPUT_PATH" ]; then
    echo "Error: Path '$INPUT_PATH' does not exist."
    exit 1
fi

# Resolve to absolute path to handle relative paths correctly
ABS_INPUT_PATH=$(realpath "$INPUT_PATH")

if [ -d "$ABS_INPUT_PATH" ]; then
    echo "Processing directory: $ABS_INPUT_PATH"
    
    # Construct find command arguments for supported extensions
    find_args=()
    for ext in "${SUPPORTED_EXTENSIONS[@]}"; do
        if [ ${#find_args[@]} -eq 0 ]; then
            find_args+=(-iname "*$ext")
        else
            find_args+=(-o -iname "*$ext")
        fi
    done
    
    # Find and process files, using print0 and read -d $'\0' for robustness
    find "$ABS_INPUT_PATH" -type f \( "${find_args[@]}" \) -print0 | while IFS= read -r -d $'\0' file_to_process; do
        process_file "$file_to_process"
    done
    echo "Finished processing directory."

elif [ -f "$ABS_INPUT_PATH" ]; then
    # Check if the single file has a supported extension
    filename_lower=$(basename "$ABS_INPUT_PATH" | tr '[:upper:]' '[:lower:]')
    file_ext_lower=""
    if [[ "$filename_lower" == *.* ]]; then # Check if there is an extension
        file_ext_lower=".${filename_lower##*.}"
    fi

    is_supported=false
    for sup_ext in "${SUPPORTED_EXTENSIONS[@]}"; do
        if [[ "$file_ext_lower" == "$sup_ext" ]]; then
            is_supported=true
            break
        fi
    done

    if [ "$is_supported" = true ]; then
        process_file "$ABS_INPUT_PATH"
        echo "Finished processing file."
    else
        echo "Error: File '$ABS_INPUT_PATH' does not appear to be a supported file type or has no extension."
        echo "Supported types are: ${SUPPORTED_EXTENSIONS[*]}"
        exit 1
    fi
else
    echo "Error: Path '$ABS_INPUT_PATH' is not a valid file or directory."
    exit 1
fi

echo "Document processing script finished."
