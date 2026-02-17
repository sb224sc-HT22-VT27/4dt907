import { describe, it, expect } from '@jest/globals';
import request from 'supertest';
import app from '../../../../src/index.js';

describe('Model Info Endpoints', () => {
    describe('GET /api/v1/model-info/champion', () => {
        it('should return 503 when MLflow is not configured', async () => {
            const response = await request(app).get('/api/v1/model-info/champion');

            // Without MLflow configured, should return error
            expect(response.status).toBe(503);
        });
    });

    describe('GET /api/v1/model-info/latest', () => {
        it('should return 503 when MLflow is not configured', async () => {
            const response = await request(app).get('/api/v1/model-info/latest');

            expect(response.status).toBe(503);
        });
    });

    describe('GET /api/v1/model-info/weakest-link/champion', () => {
        it('should return 503 when MLflow is not configured', async () => {
            const response = await request(app).get('/api/v1/model-info/weakest-link/champion');

            expect(response.status).toBe(503);
        });
    });

    describe('GET /api/v1/model-info/weakest-link/latest', () => {
        it('should return 503 when MLflow is not configured', async () => {
            const response = await request(app).get('/api/v1/model-info/weakest-link/latest');

            expect(response.status).toBe(503);
        });
    });
});

