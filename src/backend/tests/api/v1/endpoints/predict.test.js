import { describe, it, expect } from '@jest/globals';
import request from 'supertest';
import app from '../../../../src/index.js';

// Note: These tests require mocking which is complex with ES modules
// For now, testing validation and error handling without actual model loading

describe('Predict Endpoints', () => {
    describe('POST /api/v1/predict/champion', () => {
        it('should return 422 for invalid features type', async () => {
            const response = await request(app)
                .post('/api/v1/predict/champion')
                .send({ features: 'invalid' });

            expect(response.status).toBe(422);
            expect(response.body).toHaveProperty('detail');
        });

        it('should return 422 for missing features', async () => {
            const response = await request(app)
                .post('/api/v1/predict/champion')
                .send({});

            expect(response.status).toBe(422);
        });

        it('should return 503 when MLflow is not configured', async () => {
            // Without MLFLOW_TRACKING_URI set, should fail
            const response = await request(app)
                .post('/api/v1/predict/champion')
                .send({ features: [1.0, 2.0, 3.0] });

            expect(response.status).toBe(503);
            expect(response.body).toHaveProperty('detail');
        });
    });

    describe('POST /api/v1/predict/latest', () => {
        it('should return 422 for missing features', async () => {
            const response = await request(app)
                .post('/api/v1/predict/latest')
                .send({});

            expect(response.status).toBe(422);
        });

        it('should return 422 for non-array features', async () => {
            const response = await request(app)
                .post('/api/v1/predict/latest')
                .send({ features: {} });

            expect(response.status).toBe(422);
        });
    });
});

