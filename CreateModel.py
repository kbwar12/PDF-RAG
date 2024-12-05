def create_modelfile():
    # Define the modelfile content
    modelfile_content = '''FROM llama3.2

# System prompt that defines the model's behavior
SYSTEM """
You are an AI trained to extract structured information from research papers.
Your responses must always be in strict JSON format with the following structure:

{
    "abstract": "The paper's abstract content",
    "conclusion": "The paper's conclusion content",
    "keywords": "comma-separated keywords from the paper"
}

Important guidelines:
- Return only valid JSON objects
- Do not include any additional text outside the JSON structure
- Ensure proper JSON formatting and escaping of special characters
- If a section is not found, use an empty string for that field
"""

# Parameter configurations
PARAMETER temperature 0.7
PARAMETER top_p 0.95
PARAMETER stop "</s>"
PARAMETER top_k 50

# Template for handling input
TEMPLATE """
{{.System}}

Extract the required information from the following text and return it in JSON format:

{{.Prompt}}
"""
'''
    
    # Write the modelfile
    try:
        with open('Modelfile', 'w') as f:
            f.write(modelfile_content)
        print("Modelfile created successfully!")
        
        # Print instructions for using the modelfile
        print("\nTo create the model, run:")
        print("ollama create paper-extractor -f Modelfile")
        print("\nTo use the model:")
        print("ollama run paper-extractor 'your research paper text here'")
        
    except Exception as e:
        print(f"Error creating Modelfile: {str(e)}")

if __name__ == "__main__":
    create_modelfile()

# Created/Modified files during execution:
# Modelfile

