import os
import sys
# New Import for dotenv functionality
from dotenv import load_dotenv
# Import for secure key input
import getpass

# Define the environment variable name we are looking for
# We are now targeting the OpenAI key as per your request
API_KEY_ENV_VAR = "OPENAI_API_KEY"


def load_openai_api_key():
    """
    Attempts to load the API key from a .env file and environment variables.
    If not found, it prompts the user to enter the key securely.

    Returns:
        str: The loaded or entered API key.
    """
    api_key_env_var = "OPENAI_API_KEY"
    # 1. Load variables from the .env file (if it exists)
    load_dotenv()

    # 2. CORRECT WAY: Retrieve the key from the environment dictionary
    # after it has been populated by load_dotenv()
    api_key = os.getenv(api_key_env_var)

    if api_key:
        print(f"✅ Success: Loaded API Key from environment variable '{api_key_env_var}'.")
        # For security, only print the first few characters
        print(f"   Key starts with: {api_key[:4]}...")
        return api_key
    else:
        # 3. Key not found, prompt the user for secure input
        print(f"⚠️ Warning: API Key '{api_key_env_var}' not found in environment or .env file.")
        print("-" * 50)

        while True:
            # CORRECT WAY: Use getpass.getpass() to hide input
            user_input = getpass.getpass("Please enter your API Key now (input hidden): ").strip()

            if user_input:
                print("\nKey accepted. Proceeding with application logic.")
                print("-" * 50)
                return user_input
            else:
                print("Input cannot be empty. Please try again or press Ctrl+C to exit.")


# Example usage:
if __name__ == '__main__':
    key = load_openai_api_key()

    # Simple verification of the returned key
    if key:
        print(f"\nFinal retrieved key for application: {key[:4]}...")
