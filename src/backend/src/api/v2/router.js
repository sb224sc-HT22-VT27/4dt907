import express from 'express';

const router = express.Router();

router.get('/status', (req, res) => {
    res.json({ version: 'v2', status: 'ok' });
});

export default router;
