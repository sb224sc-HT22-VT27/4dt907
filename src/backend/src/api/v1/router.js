import express from 'express';
import predictRouter from './endpoints/predict.js';
import modelInfoRouter from './endpoints/modelInfo.js';
import weakestLinkRouter from './endpoints/weakestLink.js';

const router = express.Router();

router.use(predictRouter);
router.use(modelInfoRouter);
router.use(weakestLinkRouter);

export default router;
