import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import {
    getCachedCatalog,
    generateSmartRecommendations,
    enhanceWithTensorFlow,
    type RecommendationRequest
} from '@/lib/gemini/catalog-generator';
import { calculateSimilarity, getTensorFlowInfo } from '@/lib/tensorflow/recommendation-engine';

// Request validation
const RecommendationRequestSchema = z.object({
    productId: z.string(),
    strategy: z.enum(['cross-sell', 'upsell', 'bundle', 'smart']).optional(),
    useTensorFlowEnhancement: z.boolean().optional().default(false),
    context: z.string().optional(),
    customerProfile: z.object({
        purchaseHistory: z.array(z.string()).optional(),
        budget: z.number().optional(),
        preferences: z.array(z.string()).optional()
    }).optional()
});

export async function POST(request: NextRequest) {
    try {
        // Check if Gemini API key is set
        if (!process.env.GEMINI_API_KEY) {
            return NextResponse.json(
                {
                    error: 'Gemini API key not configured',
                    details: 'Please set GEMINI_API_KEY in your .env.local file'
                },
                { status: 503 }
            );
        }

        const body = await request.json();
        const params = RecommendationRequestSchema.parse(body);

        // Get Gemini-generated catalog
        const catalog = await getCachedCatalog();

        // Find target product
        const targetProduct = catalog.products.find(p => p.id === params.productId);
        if (!targetProduct) {
            return NextResponse.json(
                { error: 'Product not found in catalog' },
                { status: 404 }
            );
        }

        // Build recommendation request
        const recommendationRequest: RecommendationRequest = {
            productId: params.productId,
            strategy: params.strategy,
            context: params.context,
            customerProfile: params.customerProfile
        };

        // PRIMARY: Get Gemini AI recommendations
        let result = await generateSmartRecommendations(
            recommendationRequest,
            catalog.products
        );

        // OPTIONAL: Enhance with TensorFlow similarity scores
        if (params.useTensorFlowEnhancement) {
            try {
                result = await enhanceWithTensorFlow(
                    result,
                    calculateSimilarity,
                    targetProduct,
                    catalog.products
                );
            } catch (error) {
                console.warn('TensorFlow enhancement failed, using pure Gemini:', error);
                // Continue with pure Gemini result
            }
        }

        // Get TensorFlow backend info
        const tfInfo = getTensorFlowInfo();

        return NextResponse.json({
            targetProduct,
            recommendations: result.recommendations,
            strategy: result.strategy,
            confidence: result.confidence,
            analysis: result.analysis,
            powered: {
                gemini: true,
                tensorflow: result.tensorFlowEnhanced || false,
                backend: tfInfo.backend
            },
            catalog: {
                totalProducts: catalog.products.length,
                categories: catalog.categories,
                generatedAt: catalog.generatedAt
            }
        });

    } catch (error) {
        if (error instanceof z.ZodError) {
            return NextResponse.json(
                { error: 'Invalid parameters', details: error.issues },
                { status: 400 }
            );
        }

        console.error('Recommendation error:', error);
        return NextResponse.json(
            { error: 'Failed to generate recommendations', details: String(error) },
            { status: 500 }
        );
    }
}

// GET endpoint to list all products (from Gemini-generated catalog)
export async function GET(request: NextRequest) {
    try {
        const { searchParams } = new URL(request.url);
        const forceRefresh = searchParams.get('refresh') === 'true';

        // Check if Gemini API key is set
        if (!process.env.GEMINI_API_KEY) {
            return NextResponse.json(
                {
                    error: 'Gemini API key not configured',
                    details: 'Please set GEMINI_API_KEY in your .env.local file. Get one from https://ai.google.dev/',
                    suggestion: 'Add: GEMINI_API_KEY=your_key_here to .env.local'
                },
                { status: 503 } // Service Unavailable
            );
        }

        // Get Gemini-generated catalog
        const catalog = await getCachedCatalog(
            'electronics and accessories e-commerce',
            forceRefresh
        );

        const tfInfo = getTensorFlowInfo();

        return NextResponse.json({
            products: catalog.products,
            totalProducts: catalog.products.length,
            categories: catalog.categories,
            totalValue: catalog.totalValue,
            generatedAt: catalog.generatedAt,
            context: catalog.context,
            powered: {
                gemini: true,
                tensorflow: false
            },
            tensorFlowBackend: tfInfo.backend,
            tensorFlowReady: tfInfo.ready
        });
    } catch (error) {
        console.error('Products list error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch products', details: String(error) },
            { status: 500 }
        );
    }
}
