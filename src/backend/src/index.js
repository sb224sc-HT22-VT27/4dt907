/**
 * Express.js backend application for 4dt907 project.
 */

import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

import healthRouter from './api/health.js';
import v1Router from './api/v1/router.js';
import v2Router from './api/v2/router.js';

// Setup __dirname for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load environment variables from multiple locations
// - current working dir .env (common when running from src/backend)
// - backend/.env
// - src/.env (common when running docker-compose from src/)
dotenv.config();
dotenv.config({ path: path.join(__dirname, '..', '.env') });
dotenv.config({ path: path.join(__dirname, '..', '..', '.env') });

const app = express();

const ALLOWED_ORIGINS = [
    `http://localhost:${process.env.FRONTEND_PORT || '3030'}`,
    'http://localhost:3000',
    'http://localhost:5173',
];

const HOST_PORT = parseInt(process.env.BACKEND_PORT || '8080', 10);

// Middleware
app.use(cors({
    origin: ALLOWED_ORIGINS,
    credentials: false,
    methods: ['GET', 'POST'],
}));

app.use(express.json());

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'Backend is running',
        docs: '/docs',
        health: '/health',
        predict_champion: '/api/v1/predict/champion',
        predict_latest: '/api/v1/predict/latest',
        v2_status: '/api/v2/status',
    });
});

// Routes
app.use(healthRouter);
app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);

// Start server only if this file is run directly
if (import.meta.url === `file://${process.argv[1]}`) {
    app.listen(HOST_PORT, '0.0.0.0', () => {
        console.log(`Server running on http://0.0.0.0:${HOST_PORT}`);
    });
}

export default app;
