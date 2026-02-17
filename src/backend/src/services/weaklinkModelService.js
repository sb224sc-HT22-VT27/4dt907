/**
 * Weakest-link model service for MLflow integration
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
        return cleanUri(process.env.WEAKLINK_MODEL_URI_PROD);
    }
    if (['latest', 'dev', 'development'].includes(v)) {
        return cleanUri(process.env.WEAKLINK_MODEL_URI_DEV);
    }
    if (v === 'backup') {
        return cleanUri(process.env.WEAKLINK_MODEL_URI_BACKUP);
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

async function resolveAliasToVersionUri(modelName, alias) {
    const trackingUri = process.env.MLFLOW_TRACKING_URI;
    if (!trackingUri) {
        throw new Error('MLFLOW_TRACKING_URI is not set');
    }

    try {
        const response = await axios.get(`${trackingUri}/api/2.0/mlflow/model-versions/search`, {
            params: {
                filter: `name='${modelName}'`
            }
        });

        const versions = response.data.model_versions || [];
        if (versions.length === 0) {
            throw new Error(`No versions found for model '${modelName}'`);
        }

        const versionsSorted = versions.sort((a, b) => parseInt(b.version) - parseInt(a.version));
        const latest = versionsSorted[0];
        const secondLatest = versionsSorted.length >= 2 ? versionsSorted[1] : latest;

        const aliasLower = alias.toLowerCase().trim();

        if (['prod', 'production'].includes(aliasLower)) {
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

        return `models:/${modelName}/${latest.version}`;
    } catch (error) {
        throw new Error(`Failed to resolve alias for model ${modelName}: ${error.message}`);
    }
}

async function loadModelWithAliasFallback(uri) {
    try {
        const modelInfo = {
            uri,
            loaded: true,
            expectedFeatures: null
        };
        return { model: modelInfo, uriUsed: uri };
    } catch (error) {
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

export async function getModel(variant = 'champion') {
    initMlflow();
    
    const directUri = directUriForVariant(variant);
    if (!directUri) {
        throw new Error('Weaklink model URI is not set for this variant');
    }
    
    const cacheKey = directUri;
    if (cache.has(cacheKey)) {
        return cache.get(cacheKey);
    }
    
    const { model, uriUsed } = await loadModelWithAliasFallback(directUri);
    const result = { model, uri: uriUsed };
    cache.set(cacheKey, result);
    return result;
}

export async function expectedFeatureCount(variant = 'champion') {
    const modelInfo = await getModel(variant);
    if (!modelInfo) return null;
    return modelInfo.model.expectedFeatures;
}

/**
 * Make a prediction using the weakest-link model
 * Returns a string prediction (class label) instead of float
 */
export async function predictOne(features, variant = 'champion') {
    const modelInfo = await getModel(variant);
    if (!modelInfo) {
        throw new Error(`Weaklink model not found for variant: ${variant}`);
    }
    
    const expectedCount = await expectedFeatureCount(variant);
    if (expectedCount !== null && features.length !== expectedCount) {
        throw new Error(`Model expects ${expectedCount} features, got ${features.length}`);
    }
    
    // Placeholder prediction - should return a class label (string)
    const prediction = 'class_A';
    
    return {
        prediction,
        modelUri: modelInfo.uri
    };
}

export function clearModelCache() {
    cache.clear();
}
