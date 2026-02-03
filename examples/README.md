# API Usage Examples

This directory contains example code demonstrating how to use the 4dt907 ML prediction API.

## Python Client Example

[`api_client_example.py`](api_client_example.py) - A complete Python client showing how to:
- Connect to the API
- Check service health
- Make predictions with champion and latest models
- Handle errors gracefully

### Running the Example

1. Make sure the backend service is running:
   ```bash
   cd ../src
   docker compose up -d
   ```

2. Install requirements:
   ```bash
   pip install requests
   ```

3. Run the example:
   ```bash
   python api_client_example.py
   ```

### Integrating into Your Application

Copy the `PredictionClient` class into your application and use it like this:

```python
from api_client_example import PredictionClient

# Create client
client = PredictionClient(base_url="http://localhost:8080")

# Make predictions
result = client.predict([1.0, 2.0, 3.0], model="champion")
print(f"Prediction: {result['prediction']}")
```

## cURL Examples

### Health Check
```bash
curl http://localhost:8080/health
```

### Predict with Champion Model
```bash
curl -X POST http://localhost:8080/api/v1/predict/champion \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

### Predict with Latest Model
```bash
curl -X POST http://localhost:8080/api/v1/predict/latest \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0]}'
```

## JavaScript/Node.js Example

```javascript
async function predict(features, model = 'champion') {
  const response = await fetch(`http://localhost:8080/api/v1/predict/${model}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features })
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail);
  }
  
  return await response.json();
}

// Usage
predict([1.0, 2.0, 3.0], 'champion')
  .then(result => console.log('Prediction:', result.prediction))
  .catch(err => console.error('Error:', err));
```

## See Also

- [Deployment Guide](../DEPLOYMENT.md) - Complete setup and deployment instructions
- [Backend README](../src/backend/README.md) - Backend API documentation
- [API Interactive Docs](http://localhost:8080/docs) - When service is running
