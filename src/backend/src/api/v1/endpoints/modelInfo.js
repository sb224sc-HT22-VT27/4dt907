import express from 'express';
import * as modelService from '../../../services/modelService.js';
import * as weaklinkModelService from '../../../services/weaklinkModelService.js';

const router = express.Router();

router.get('/model-info/latest', async (req, res) => {
    try {
        const modelInfo = await modelService.getModel('latest');
        const expectedFeatures = await modelService.expectedFeatureCount('latest');
        res.json({
            variant: 'latest',
            model_uri: modelInfo.uri,
            expected_features: expectedFeatures
        });
    } catch (error) {
        console.error('Failed to get model info:', error);
        res.status(503).json({ detail: error.message });
    }
});

router.get('/model-info/champion', async (req, res) => {
    try {
        const modelInfo = await modelService.getModel('champion');
        const expectedFeatures = await modelService.expectedFeatureCount('champion');
        res.json({
            variant: 'champion',
            model_uri: modelInfo.uri,
            expected_features: expectedFeatures
        });
    } catch (error) {
        console.error('Failed to get model info:', error);
        res.status(503).json({ detail: error.message });
    }
});

router.get('/model-info/weakest-link/latest', async (req, res) => {
    try {
        const modelInfo = await weaklinkModelService.getModel('latest');
        const expectedFeatures = await weaklinkModelService.expectedFeatureCount('latest');
        res.json({
            variant: 'latest',
            model_uri: modelInfo.uri,
            expected_features: expectedFeatures
        });
    } catch (error) {
        console.error('Failed to get weakest-link model info:', error);
        res.status(503).json({ detail: error.message });
    }
});

router.get('/model-info/weakest-link/champion', async (req, res) => {
    try {
        const modelInfo = await weaklinkModelService.getModel('champion');
        const expectedFeatures = await weaklinkModelService.expectedFeatureCount('champion');
        res.json({
            variant: 'champion',
            model_uri: modelInfo.uri,
            expected_features: expectedFeatures
        });
    } catch (error) {
        console.error('Failed to get weakest-link model info:', error);
        res.status(503).json({ detail: error.message });
    }
});

export default router;
