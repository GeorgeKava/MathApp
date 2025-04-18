from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
    SearchIndex,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters
)
from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI
import os
import logging  # Logging facility for Python
from dotenv import load_dotenv, set_key
from IPython.display import Image, display, Audio, Markdown
import base64  # Base16, Base32, Base64, Base85 Data Encodings
import json
import re  # Regular expression operations
from mimetypes import guess_type
import uuid  # Add this import

load_dotenv()

search_service_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_api_key = os.getenv("AZURE_SEARCH_KEY")


azure_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")
model = os.getenv('AZURE_OPENAI_MODEL')

embedding_model = "text-embedding-3-small"

index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')

env_file_path = '.env'

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{azure_endpoint}/openai/deployments/{model}",
)

embedding_client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    base_url=f"{azure_endpoint}/openai/deployments/{embedding_model}",
)

search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)

index_client = SearchIndexClient(
    endpoint=search_service_endpoint,
    credential=AzureKeyCredential(search_api_key)
)

index_schema = SearchIndex(
    name=index_name,
    fields=[
        SimpleField(name="chunk_id", type="Edm.String", sortable=True,
                    filterable=True, facetable=True, key=True),
        SearchableField(name="page_content", type="Edm.String",
                        searchable=True, retrievable=True),
        SearchableField(name="filename",
                        type="Edm.String", searchable=True, retrievable=True),
        SearchableField(name="title",
                type="Edm.String", searchable=True, retrievable=True),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=int(1536),
            vector_search_profile_name="myHnswProfile",
        )
    ]
)

def create_vector_search():
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
                vectorizer="myVectorizer"
            )
        ],
        vectorizers=[
            AzureOpenAIVectorizer(
                name="myVectorizer",
                azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                    resource_uri=azure_endpoint,
                    deployment_id=embedding_model,
                    model_name=embedding_model,
                    api_key=api_key
                )
            )
        ]
    )
    return vector_search

def create_index():
    try:
        vector_search = create_vector_search()
        index_schema.vector_search = vector_search
        index_client.create_index(index_schema)
        logging.info(f"Index '{index_name}' created successfully.")
    except Exception as e:
        logging.error(f"Failed to create index: {e}")
        
def local_image_to_data_url(image_path):  # Get the url of a local image
    mime_type, _ = guess_type(image_path)

    if (mime_type is None):
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(
            image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{base64_encoded_data}"

def process_img_llm(img_name):
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{azure_endpoint}/openai/deployments/{model}",
    )
    prompt = """You are a expert math assistant. Analyze the image provided and extract any math problems.
                Solve the math problems and provide detailed solutions.
                If there are multiple problems, solve each one and provide solutions in a JSON format.
                JSON Format:
                [
                    {
                        "problem": "2 + 2",
                        "solution": "The solution is 4."
                    },
                    {
                        "problem": "5 * 3",
                        "solution": "The solution is 15."
                    }
                ]
            """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to analyse images.",
            },
            {
                "role": "user",
                "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(img_name)},
                        },
                ],
            },
        ],
        max_tokens=2000,
        temperature=0.0,
    )
    response_content = response.choices[0].message.content.strip()
    logging.debug(f"Response content: {response_content}")

    # Attempt to parse the response content as JSON
    json_match = re.search(r'\{.*\}', response_content,
                           re.DOTALL)  # Match a JSON object
    if json_match:
        # Extract the JSON object from the response content
        json_str = json_match.group(0)
        # Remove the JSON object from the response content
        summary = response_content.replace(json_str, '').strip()
        # Remove 'json[]' from the summary
        summary = re.sub(r'json\s*\[\s*\]', '', summary).strip()
        # Remove code blocks from the summary
        summary = summary.replace('```', '').strip()
        try:
            response_json = json.loads(json_str)
            logging.debug(f"Parsed JSON: {response_json}")
            # Ensure the response contains the required keys
            required_keys = ["problem", "solution"]
            if not all(key in response_json for key in required_keys):
                raise ValueError(
                    "Response JSON does not contain all required keys")
            # Create a formatted summary
            formatted_summary = (
                f"Problem: {response_json['problem']}<br>"
                f"Solution: {response_json['solution']}<br>"
                f"{summary}"
            )
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(
                f"Failed to parse response as JSON or missing keys: {e}")
            response_json = {
                "error": "Failed to parse response as JSON or missing keys",
                "response": response_content
            }
            formatted_summary = response_content
    else:
        logging.error("No JSON object found in the response content")
        response_json = {
            "error": "No JSON object found in the response content",
            "response": response_content
        }
        formatted_summary = response_content

    # Read existing data from the JSON file
    json_filename = 'math.json'
    try:
        with open(json_filename, 'r') as json_file:  # Open the JSON file for reading
            # Load the existing data from the JSON file
            existing_data = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file doesn't exist or is empty, initialize existing data as an empty list
        existing_data = []

    # Append the new data to the existing data
    if isinstance(existing_data, list):
        existing_data.append(response_json)
    else:
        existing_data = [existing_data, response_json]

    # Save the updated data back to the JSON file
    with open(json_filename, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    return {
        "formatted_summary": formatted_summary
    }

def format_result_for_display(result):
    # Example: Convert the result dictionary into Markdown with LaTeX
    formatted_result = "### Extracted Problems and Solutions:\n\n"
    for i, item in enumerate(result, start=1):
        problem = item.get("problem", "No problem provided")
        solution = item.get("solution", "No solution provided")
        formatted_result += f"**Problem {i}:**\n"
        formatted_result += f"\\[\n{problem}\n\\]\n"
        formatted_result += f"**Solution:**\n"
        formatted_result += f"\\[\n{solution}\n\\]\n\n"
    return formatted_result

def process_text_math_problem(problem):
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{azure_endpoint}/openai/deployments/{model}",
    )
    prompt = f"""You are an expert math assistant. Solve the following problem:
                {problem}
                Provide a detailed solution in plain text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for solving math problems.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=2000,
        temperature=0.0,
    )
    response_content = response.choices[0].message.content.strip()
    return response_content

def process_img_llm_chemistry(img_name):
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{azure_endpoint}/openai/deployments/{model}",
    )
    prompt = """You are an expert chemistry assistant. Analyze the image provided and extract any chemistry problems.
                Solve the chemistry problems and provide detailed solutions.
                If the problem is a multiple-choice question, first state the correct answer and then provide a detailed explanation.
                Return the results in the following JSON format:
                [
                    {
                        "problem": "What is the molar mass of H2O?",
                        "solution": "The molar mass of H2O is 18.015 g/mol."
                    },
                    {
                        "problem": "Balance the equation: H2 + O2 -> H2O",
                        "solution": "The balanced equation is 2H2 + O2 -> 2H2O."
                    },
                    {
                        "problem": "Which of the following is the molar mass of H2O? (a) 18.015 g/mol (b) 20 g/mol (c) 22 g/mol",
                        "solution": "The correct answer is (a) 18.015 g/mol. The molar mass of H2O is calculated as follows: 2(1.008) + 15.999 = 18.015 g/mol."
                    }
                ]
            """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to analyze chemistry problems.",
            },
            {
                "role": "user",
                "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(img_name)},
                        },
                ],
            },
        ],
        max_tokens=2000,
        temperature=0.0,
    )
    response_content = response.choices[0].message.content.strip()
    logging.debug(f"Response content: {response_content}")

    # Attempt to parse the response content as JSON
    try:
        response_json = json.loads(response_content)
        formatted_summary = format_result_for_display(response_json.get("detailed_solutions", []))
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse response as JSON: {e}")
        response_json = {"error": "Failed to parse response as JSON"}
        formatted_summary = response_content

    return {
        "formatted_summary": formatted_summary
    }

def process_text_chemistry_problem(problem):
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{azure_endpoint}/openai/deployments/{model}",
    )
    prompt = f"""You are an expert chemistry assistant. Solve the following problem:
                {problem}
                Provide a detailed solution in plain text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for solving chemistry problems.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=2000,
        temperature=0.0,
    )
    response_content = response.choices[0].message.content.strip()
    return response_content

def process_img_llm_physics(img_name):
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{azure_endpoint}/openai/deployments/{model}",
    )
    prompt = """You are an expert physics assistant. Analyze the image provided and extract any physics problems.
                Solve the physics problems and provide detailed solutions.
                If the problem is a multiple-choice question, first state the correct answer and then provide a detailed explanation.
                Return the results in the following JSON format:
                [
                    {
                        "problem": "What is the acceleration of an object with a mass of 5 kg and a force of 20 N applied to it?",
                        "solution": "The acceleration is calculated using Newton's second law: F = ma. Rearranging, a = F/m = 20 N / 5 kg = 4 m/s²."
                    },
                    {
                        "problem": "What is the gravitational potential energy of a 2 kg object raised to a height of 10 m? (g = 9.8 m/s²)",
                        "solution": "The gravitational potential energy is calculated as U = mgh. Substituting, U = 2 kg × 9.8 m/s² × 10 m = 196 J."
                    },
                    {
                        "problem": "Which of the following is the correct formula for kinetic energy? (a) KE = 1/2 mv² (b) KE = mv² (c) KE = 1/2 mv",
                        "solution": "The correct answer is (a) KE = 1/2 mv². Kinetic energy is defined as the energy of motion, and the formula is derived from the work-energy principle."
                    }
                ]
            """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant to analyze physics problems.",
            },
            {
                "role": "user",
                "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": local_image_to_data_url(img_name)},
                        },
                ],
            },
        ],
        max_tokens=2000,
        temperature=0.0,
    )
    response_content = response.choices[0].message.content.strip()
    logging.debug(f"Response content: {response_content}")

    # Attempt to parse the response content as JSON
    try:
        response_json = json.loads(response_content)
        formatted_summary = format_result_for_display(response_json.get("detailed_solutions", []))
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse response as JSON: {e}")
        response_json = {"error": "Failed to parse response as JSON"}
        formatted_summary = response_content

    return {
        "formatted_summary": formatted_summary
    }

def process_text_physics_problem(problem):
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{azure_endpoint}/openai/deployments/{model}",
    )
    prompt = f"""You are an expert physics assistant. Solve the following problem:
                {problem}
                Provide a detailed solution in plain text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for solving physics problems.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=2000,
        temperature=0.0,
    )
    response_content = response.choices[0].message.content.strip()
    return response_content


if __name__ == "__main__":
    create_index()