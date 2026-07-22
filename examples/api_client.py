"""HTTP API Client example communicating with FoundationLM FastAPI server."""

from sdk.foundationlm.client import FoundationLMClient


def main():
    # Make sure uvicorn api.main:app is running on localhost:8000
    client = FoundationLMClient(base_url="http://127.0.0.1:8000")

    # Health check
    try:
        health = client.health()
        print(f"Server Health: {health}")
    except Exception as e:
        print(f"Could not connect to API server: {e}")
        return

    # Generation request
    output = client.generate("The future of computing is", max_tokens=32, temperature=0.7)
    print(f"Generated text: {output}")


if __name__ == "__main__":
    main()
