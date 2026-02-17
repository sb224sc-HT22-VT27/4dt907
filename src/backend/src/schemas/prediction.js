import Joi from 'joi';

export const PredictRequestSchema = Joi.object({
    features: Joi.array().items(Joi.number()).required()
});

export const PredictResponseSchema = Joi.object({
    prediction: Joi.number().required(),
    model_uri: Joi.string().required()
});

export const WeakestLinkResponseSchema = Joi.object({
    prediction: Joi.string().required(),
    model_uri: Joi.string().required()
});
