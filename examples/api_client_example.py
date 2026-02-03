#!/usr/bin/env python3
"""
Example script demonstrating how to use the 4dt907 ML prediction API.

This script shows how to:
1. Send feature data to the API
2. Receive predictions from the champion or latest model
3. Handle errors appropriately

Usage:
    python api_client_example.py
    
Requirements:
    pip install requests
"""

import requests
import sys
from typing import List


class PredictionClient:
    """Client for interacting with the 4dt907 prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        """
        Initialize the prediction client.
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8080)
        """
        self.base_url = base_url.rstrip('/')
        
    def health_check(self) -> bool:
        """
        Check if the API is healthy and responding.
        
        Returns:
            True if the API is healthy, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
            data = response.json()
            return data.get("status") == "ok"
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def predict(self, features: List[float], model: str = "champion") -> dict:
        """
        Get a prediction from the API.
        
        Args:
            features: List of feature values
            model: Model to use ("champion" or "latest")
            
        Returns:
            Dictionary with prediction and model_uri
            
        Raises:
            ValueError: If model parameter is invalid
            RuntimeError: If the API request fails
        """
        if model not in ["champion", "latest"]:
            raise ValueError(f"Model must be 'champion' or 'latest', got: {model}")
        
        endpoint = f"{self.base_url}/api/v1/predict/{model}"
        
        try:
            response = requests.post(
                endpoint,
                json={"features": features},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Try to get JSON response
            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError:
                data = {}
            
            # Handle error responses
            if not response.ok:
                error_detail = data.get("detail", f"HTTP {response.status_code}")
                raise RuntimeError(f"Prediction failed: {error_detail}")
            
            return data
            
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Could not connect to the API. "
                "Make sure the backend is running at " + self.base_url
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("Request timed out")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")


def main():
    """Main function demonstrating API usage."""
    
    # Initialize client
    client = PredictionClient(base_url="http://localhost:8080")
    
    print("=" * 60)
    print("4dt907 ML Prediction API - Example Client")
    print("=" * 60)
    print()
    
    # Check API health
    print("1. Checking API health...")
    if not client.health_check():
        print("   ❌ API is not responding. Make sure the backend is running.")
        print("   Run: cd src && docker compose up -d")
        sys.exit(1)
    print("   ✅ API is healthy")
    print()
    
    # Example 1: Predict with champion model
    print("2. Getting prediction from CHAMPION model...")
    try:
        features = [1.0, 2.0, 3.0]  # Replace with your actual feature values
        result = client.predict(features, model="champion")
        print(f"   Features: {features}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Model URI: {result['model_uri']}")
        print("   ✅ Success")
    except RuntimeError as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Example 2: Predict with latest model
    print("3. Getting prediction from LATEST model...")
    try:
        features = [1.5, 2.5, 3.5]  # Replace with your actual feature values
        result = client.predict(features, model="latest")
        print(f"   Features: {features}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Model URI: {result['model_uri']}")
        print("   ✅ Success")
    except RuntimeError as e:
        print(f"   ❌ Error: {e}")
    print()
    
    # Example 3: Demonstrate error handling
    print("4. Demonstrating error handling (invalid features)...")
    try:
        # This will fail if the model expects a different number of features
        features = [1.0]  # Wrong number of features
        result = client.predict(features, model="champion")
        print(f"   Prediction: {result['prediction']}")
    except RuntimeError as e:
        print(f"   ✅ Error handled correctly: {e}")
    print()
    
    print("=" * 60)
    print("Examples completed!")
    print()
    print("Next steps:")
    print("  - Modify the feature values to match your model's inputs")
    print("  - Integrate this client into your application")
    print("  - See DEPLOYMENT.md for more information")
    print("=" * 60)


if __name__ == "__main__":
    main()
