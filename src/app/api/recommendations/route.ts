import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { getSmartRecommendations, DEMO_PRODUCTS, getTensorFlowInfo } from '@/lib/tensorflow/recommendation-engine';
import { analyzeRecommendations, quickAnalyzeRecommendations } from '@/lib/gemini/recommendation-analysis';

// Request validation
const RecommendationRequestSchema = z.object({
    productId: z.string(),
    strategy: z.enum(['similar', 'complementary', 'upsell', 'mixed']).optional(),
    useGeminiAnalysis: z.boolean().optional().default(false),
});

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const params = RecommendationRequestSchema.parse(body);

        // Get product from catalog
        const targetProduct = DEMO_PRODUCTS.find(p => p.id === params.productId);
        if (!targetProduct) {
            return NextResponse.json(
                { error: 'Product not found' },
                { status: 404 }
            );
        }

        // Get TensorFlow recommendations
        const tfResult = await getSmartRecommendations(
            params.productId,
            params.strategy || 'mixed'
        );

        // Get Gemini analysis
        let analysis;
        if (params.useGeminiAnalysis) {
            try {
                // Try full Gemini analysis (may fail without API key)
                analysis = await analyzeRecommendations(
                    targetProduct,
                    tfResult.recommendations,
                    true // Enable code execution
                );
            } catch (error) {
                console.warn('Gemini analysis failed, using quick analysis:', error);
                analysis = quickAnalyzeRecommendations(
                    targetProduct,
                    tfResult.recommendations
                );
            }
        } else {
            // Use quick local analysis
            analysis = quickAnalyzeRecommendations(
                targetProduct,
                tfResult.recommendations
            );
        }

        // Get TensorFlow backend info
        const tfInfo = getTensorFlowInfo();

        return NextResponse.json({
            targetProduct,
            recommendations: tfResult.recommendations,
            strategy: tfResult.strategy,
            tfConfidence: tfResult.confidence,
            analysis,
            tensorFlowBackend: tfInfo.backend,
            powered: {
                tensorflow: true,
                gemini: params.useGeminiAnalysis,
                backend: tfInfo.backend
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
            { error: 'Failed to generate recommendations' },
            { status: 500 }
        );
    }
}

// GET endpoint to list all products
export async function GET() {
    try {
        const tfInfo = getTensorFlowInfo();

        return NextResponse.json({
            products: DEMO_PRODUCTS,
            totalProducts: DEMO_PRODUCTS.length,
            categories: [...new Set(DEMO_PRODUCTS.map(p => p.category))],
            tensorFlowBackend: tfInfo.backend,
            tensorFlowReady: tfInfo.ready
        });
    } catch (error) {
        console.error('Products list error:', error);
        return NextResponse.json(
            { error: 'Failed to fetch products' },
            { status: 500 }
        );
    }
}
