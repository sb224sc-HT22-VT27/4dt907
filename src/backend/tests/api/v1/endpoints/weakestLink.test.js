import { describe, it, expect } from '@jest/globals';
import request from 'supertest';
import app from '../../../../src/index.js';

describe('Weakest-Link Endpoints', () => {
    describe('POST /api/v1/weakest-link/champion', () => {
        it('should return 422 for invalid input', async () => {
            const response = await request(app)
                .post('/api/v1/weakest-link/champion')
                .send({ features: 'not-an-array' });

            expect(response.status).toBe(422);
        });

        it('should return 503 when MLflow is not configured', async () => {
            const response = await request(app)
                .post('/api/v1/weakest-link/champion')
                .send({ features: [1.0, 2.0, 3.0, 4.0] });

            expect(response.status).toBe(503);
        });
    });

    describe('POST /api/v1/weakest-link/latest', () => {
        it('should return 422 for missing features', async () => {
            const response = await request(app)
                .post('/api/v1/weakest-link/latest')
                .send({});

            expect(response.status).toBe(422);
        });

        it('should return 503 when MLflow is not configured', async () => {
            const response = await request(app)
                .post('/api/v1/weakest-link/latest')
                .send({ features: [5.0, 6.0, 7.0, 8.0] });

            expect(response.status).toBe(503);
        });
    });
});

