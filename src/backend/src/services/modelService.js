/**
 * Model service for MLflow integration
 * 
 * This is an Express.js implementation that uses MLflow REST API
 * instead of the Python SDK. For production use with actual ML models,
 * consider using MLflow Model Serving or a Python microservice.
 */

import axios from 'axios';

const cache = new Map();

function cleanUri(value) {
    if (!value) return null;
    return value.trim().replace(/^["']|["']$/g, '');
}

function directUriForVariant(variant) {
    const v = (variant || '').toLowerCase().trim();
    if (['champion', 'best', 'prod', 'production'].includes(v)) {
        return cleanUri(process.env.MODEL_URI_PROD);
    }
    if (['latest', 'dev', 'development'].includes(v)) {
        return cleanUri(process.env.MODEL_URI_DEV);
    }
    if (v === 'backup') {
        return cleanUri(process.env.MODEL_URI_BACKUP);
    }
    return null;
}

function initMlflow() {
    const uri = process.env.MLFLOW_TRACKING_URI;
    if (!uri) {
        throw new Error('MLFLOW_TRACKING_URI is not set');
    }
    return uri;
}

function isModelsAliasUri(uri) {
    return (
        uri.startsWith('models:/') &&
        uri.includes('@') &&
        !uri.replace('models:/', '').split('@')[0].includes('/')
    );
}

function parseModelsAliasUri(uri) {
    const tail = uri.replace('models:/', '');
    const [name, alias] = tail.split('@');
    if (!name.trim() || !alias.trim()) {
        throw new Error(`Invalid models alias uri: ${uri}`);
    }
    return { name: name.trim(), alias: alias.trim() };
}

/**
 * Resolve alias to version URI using MLflow REST API
 * This is a fallback for registries that don't support alias lookup
 */
async function resolveAliasToVersionUri(modelName, alias) {
    const trackingUri = process.env.MLFLOW_TRACKING_URI;
    if (!trackingUri) {
        throw new Error('MLFLOW_TRACKING_URI is not set');
    }

    try {
        // Call MLflow REST API to search model versions
        const response = await axios.get(`${trackingUri}/api/2.0/mlflow/model-versions/search`, {
            params: {
                filter: `name='${modelName}'`
            }
        });

        const versions = response.data.model_versions || [];
        if (versions.length === 0) {
            throw new Error(`No versions found for model '${modelName}'`);
        }

        // Sort by version number
        const versionsSorted = versions.sort((a, b) => parseInt(b.version) - parseInt(a.version));
        const latest = versionsSorted[0];
        const secondLatest = versionsSorted.length >= 2 ? versionsSorted[1] : latest;

        const aliasLower = alias.toLowerCase().trim();

        if (['prod', 'production'].includes(aliasLower)) {
            // Prefer Production stage if available
            const prodCandidates = versionsSorted.filter(
                mv => (mv.current_stage || '').toLowerCase() === 'production'
            );
            const chosen = prodCandidates.length > 0 ? prodCandidates[0] : latest;
            return `models:/${modelName}/${chosen.version}`;
        }

        if (['dev', 'latest'].includes(aliasLower)) {
            return `models:/${modelName}/${latest.version}`;
        }

        if (aliasLower === 'backup') {
            return `models:/${modelName}/${secondLatest.version}`;
        }

        // Default to latest
        return `models:/${modelName}/${latest.version}`;
    } catch (error) {
        throw new Error(`Failed to resolve alias for model ${modelName}: ${error.message}`);
    }
}

/**
 * Load model with alias fallback
 * Returns model info and URI used
 */
async function loadModelWithAliasFallback(uri) {
    const trackingUri = process.env.MLFLOW_TRACKING_URI;
    
    try {
        // For models:/ URIs, we'll store metadata
        // In production, you would call MLflow Model Serving API here
        const modelInfo = {
            uri,
            loaded: true,
            expectedFeatures: null // Would be populated from model metadata
        };
        return { model: modelInfo, uriUsed: uri };
    } catch (error) {
        // If it's an alias URI and failed, try resolving
        if (isModelsAliasUri(uri) && error.message.includes('INVALID_PARAMETER_VALUE')) {
            const { name, alias } = parseModelsAliasUri(uri);
            const resolvedUri = await resolveAliasToVersionUri(name, alias);
            const modelInfo = {
                uri: resolvedUri,
                loaded: true,
                expectedFeatures: null
            };
            return { model: modelInfo, uriUsed: resolvedUri };
        }
        throw error;
    }
}

/**
 * Get model for a variant (champion, latest, etc.)
 */
export async function getModel(variant = 'champion') {
    initMlflow();
    
    const directUri = directUriForVariant(variant);
    if (!directUri) {
        return null;
    }
    
    const cacheKey = directUri;
    if (cache.has(cacheKey)) {
        return cache.get(cacheKey);
    }
    
    try {
        const { model, uriUsed } = await loadModelWithAliasFallback(directUri);
        const result = { model, uri: uriUsed };
        cache.set(cacheKey, result);
        return result;
    } catch (error) {
        console.error(`Failed to load model for variant ${variant}:`, error.message);
        return null;
    }
}

/**
 * Get expected feature count for a model variant
 * Note: In production, this would query the model's metadata
 */
export async function expectedFeatureCount(variant = 'champion') {
    const modelInfo = await getModel(variant);
    if (!modelInfo) return null;
    
    // This would be retrieved from actual model metadata
    // For now, returning null (unknown)
    return modelInfo.model.expectedFeatures;
}

/**
 * Make a prediction using the model
 * 
 * Note: This is a placeholder implementation. For production:
 * - Use MLflow Model Serving REST API
 * - Or call a Python microservice that handles predictions
 */
export async function predictOne(features, variant = 'champion') {
    const modelInfo = await getModel(variant);
    if (!modelInfo) {
        throw new Error(`Model not found for variant: ${variant}`);
    }
    
    const expectedCount = await expectedFeatureCount(variant);
    if (expectedCount !== null && features.length !== expectedCount) {
        throw new Error(`Model expects ${expectedCount} features, got ${features.length}`);
    }
    
    // In production, you would:
    // 1. Call MLflow Model Serving REST API
    // 2. Or send request to Python microservice
    // Example:
    // const response = await axios.post(`${MLFLOW_SERVING_URI}/invocations`, {
    //     dataframe_records: [features]
    // });
    // return response.data.predictions[0];
    
    // Placeholder prediction
    const prediction = 0.5;
    
    return {
        prediction,
        modelUri: modelInfo.uri
    };
}

/**
 * Clear the model cache
 */
export function clearModelCache() {
    cache.clear();
}
