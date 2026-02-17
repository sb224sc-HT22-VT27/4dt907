import request from 'supertest';
import app from '../src/index.js';

describe('Main Application Tests', () => {
    describe('GET /', () => {
        it('should return backend info with status 200', async () => {
            const response = await request(app).get('/');
            
            expect(response.status).toBe(200);
            expect(response.body).toHaveProperty('message', 'Backend is running');
            expect(response.body).toHaveProperty('docs', '/docs');
            expect(response.body).toHaveProperty('health', '/health');
            expect(response.body).toHaveProperty('predict_champion', '/api/v1/predict/champion');
            expect(response.body).toHaveProperty('predict_latest', '/api/v1/predict/latest');
            expect(response.body).toHaveProperty('v2_status', '/api/v2/status');
        });
    });

    describe('GET /health', () => {
        it('should return health status with status 200', async () => {
            const response = await request(app).get('/health');
            
            expect(response.status).toBe(200);
            expect(response.body).toEqual({ status: 'ok' });
        });
    });

    describe('GET /api/v2/status', () => {
        it('should return v2 status', async () => {
            const response = await request(app).get('/api/v2/status');
            
            expect(response.status).toBe(200);
            expect(response.body).toEqual({ version: 'v2', status: 'ok' });
        });
    });
});
