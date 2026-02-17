import express from 'express';
import { PredictRequestSchema } from '../../../schemas/prediction.js';
import * as weaklinkModelService from '../../../services/weaklinkModelService.js';

const router = express.Router();

router.post('/weakest-link/champion', async (req, res) => {
    try {
        // Validate request
        const { error, value } = PredictRequestSchema.validate(req.body);
        if (error) {
            return res.status(422).json({ detail: error.details[0].message });
        }

        const { prediction, modelUri } = await weaklinkModelService.predictOne(value.features, 'champion');
        res.json({ prediction, model_uri: modelUri });
    } catch (error) {
        if (error.message.includes('expects') && error.message.includes('features')) {
            return res.status(422).json({ detail: error.message });
        }
        console.error('Weakest-link prediction failed:', error);
        res.status(503).json({ detail: `${error.constructor.name}: ${error.message}` });
    }
});

router.post('/weakest-link/latest', async (req, res) => {
    try {
        // Validate request
        const { error, value } = PredictRequestSchema.validate(req.body);
        if (error) {
            return res.status(422).json({ detail: error.details[0].message });
        }

        const { prediction, modelUri } = await weaklinkModelService.predictOne(value.features, 'latest');
        res.json({ prediction, model_uri: modelUri });
    } catch (error) {
        if (error.message.includes('expects') && error.message.includes('features')) {
            return res.status(422).json({ detail: error.message });
        }
        console.error('Weakest-link prediction failed:', error);
        res.status(503).json({ detail: `${error.constructor.name}: ${error.message}` });
    }
});

export default router;
