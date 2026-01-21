/**
 * Gemini AI Product Catalog Generator
 * Dynamically generates product catalogs based on business context
 * PRIMARY AI engine for SellerOps product intelligence
 */

import { GoogleGenAI } from '@google/genai';

// === MODELS ===
const PRIMARY_MODEL = 'gemini-2.5-flash-lite';
const FALLBACK_MODEL = 'gemini-2.5-flash-lite';

// === TYPES ===
export interface Product {
    id: string;
    name: string;
    category: string;
    price: number;
    keywords: string[];
    description: string;
    specifications?: Record<string, string>;
    targetAudience?: string[];
    seasonality?: 'ALL_YEAR' | 'SEASONAL' | 'TRENDING';
}

export interface ProductCatalog {
    products: Product[];
    categories: string[];
    totalValue: number;
    generatedAt: Date;
    context: string;
}

export interface RecommendationRequest {
    productId: string;
    context?: string;
    customerProfile?: {
        purchaseHistory?: string[];
        budget?: number;
        preferences?: string[];
    };
    strategy?: 'cross-sell' | 'upsell' | 'bundle' | 'smart';
}

export interface GeminiRecommendation {
    product: Product;
    score: number;
    reason: string;
    insights: string[];
    bundleCompatibility?: number;
}

export interface RecommendationResult {
    recommendations: GeminiRecommendation[];
    strategy: string;
    confidence: number;
    analysis: {
        summary: string;
        insights: Array<{
            type: 'CROSS_SELL' | 'UPSELL' | 'BUNDLE' | 'TRENDING';
            title: string;
            description: string;
            actionable: boolean;
        }>;
        bundleOpportunities: Array<{
            products: string[];
            discount: number;
            expectedValue: number;
            reason: string;
        }>;
        pricingStrategy: string;
        expectedImpact: {
            revenueIncrease: string;
            conversionBoost: string;
        };
    };
    tensorFlowEnhanced?: boolean; // Optional TF enhancement flag
}

// === JSON SCHEMAS ===
const ProductSchema = {
    type: 'object',
    properties: {
        id: { type: 'string' },
        name: { type: 'string' },
        category: { type: 'string' },
        price: { type: 'number' },
        keywords: { type: 'array', items: { type: 'string' } },
        description: { type: 'string' },
        targetAudience: { type: 'array', items: { type: 'string' } },
        seasonality: {
            type: 'string',
            enum: ['ALL_YEAR', 'SEASONAL', 'TRENDING']
        }
    },
    required: ['id', 'name', 'category', 'price', 'keywords', 'description']
};

const CatalogSchema = {
    type: 'object',
    properties: {
        products: {
            type: 'array',
            items: ProductSchema
        },
        categories: {
            type: 'array',
            items: { type: 'string' }
        }
    },
    required: ['products', 'categories']
};

const RecommendationSchema = {
    type: 'object',
    properties: {
        recommendations: {
            type: 'array',
            items: {
                type: 'object',
                properties: {
                    productId: { type: 'string' },
                    score: { type: 'number', minimum: 0, maximum: 100 },
                    reason: { type: 'string' },
                    insights: { type: 'array', items: { type: 'string' } },
                    bundleCompatibility: { type: 'number', minimum: 0, maximum: 100 }
                },
                required: ['productId', 'score', 'reason', 'insights']
            }
        },
        strategy: { type: 'string' },
        confidence: { type: 'number', minimum: 0, maximum: 100 },
        analysis: {
            type: 'object',
            properties: {
                summary: { type: 'string' },
                insights: {
                    type: 'array',
                    items: {
                        type: 'object',
                        properties: {
                            type: {
                                type: 'string',
                                enum: ['CROSS_SELL', 'UPSELL', 'BUNDLE', 'TRENDING']
                            },
                            title: { type: 'string' },
                            description: { type: 'string' },
                            actionable: { type: 'boolean' }
                        },
                        required: ['type', 'title', 'description', 'actionable']
                    }
                },
                bundleOpportunities: {
                    type: 'array',
                    items: {
                        type: 'object',
                        properties: {
                            products: { type: 'array', items: { type: 'string' } },
                            discount: { type: 'number' },
                            expectedValue: { type: 'number' },
                            reason: { type: 'string' }
                        },
                        required: ['products', 'discount', 'expectedValue', 'reason']
                    }
                },
                pricingStrategy: { type: 'string' },
                expectedImpact: {
                    type: 'object',
                    properties: {
                        revenueIncrease: { type: 'string' },
                        conversionBoost: { type: 'string' }
                    },
                    required: ['revenueIncrease', 'conversionBoost']
                }
            },
            required: ['summary', 'insights', 'bundleOpportunities', 'pricingStrategy', 'expectedImpact']
        }
    },
    required: ['recommendations', 'strategy', 'confidence', 'analysis']
};

// === GEMINI CLIENT ===
let aiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
    if (!aiClient) {
        const apiKey = process.env.GEMINI_API_KEY;
        if (!apiKey) {
            throw new Error('GEMINI_API_KEY environment variable is not set');
        }
        aiClient = new GoogleGenAI({ apiKey });
    }
    return aiClient;
}

// === CATALOG GENERATION ===

/**
 * Generate a product catalog using Gemini AI
 * Can be customized for different business domains
 */
export async function generateProductCatalog(
    businessContext: string = 'electronics and accessories e-commerce',
    productCount: number = 10,
    useCodeExecution: boolean = true
): Promise<ProductCatalog> {
    const client = getGeminiClient();

    const prompt = `You are a product catalog generator for a ${businessContext} business.

Generate ${productCount} realistic products with pricing, keywords, and descriptions.
Return structured JSON matching the schema.`;

    // Note: codeExecution tool cannot be used with structured JSON output
    const config: any = {
        responseMimeType: 'application/json',
        responseJsonSchema: CatalogSchema,
    };

    try {
        const response = await client.models.generateContent({
            model: PRIMARY_MODEL,
            contents: prompt,
            config
        });

        const responseText = response.text || '{}';
        const catalogData = JSON.parse(responseText);

        return {
            products: catalogData.products,
            categories: catalogData.categories,
            totalValue: catalogData.products.reduce((sum: number, p: Product) => sum + p.price, 0),
            generatedAt: new Date(),
            context: businessContext
        };

    } catch (error) {
        console.error('Gemini catalog generation error:', error);

        // Fallback to simpler model
        try {
            const fallbackResponse = await client.models.generateContent({
                model: FALLBACK_MODEL,
                contents: prompt,
                config: {
                    responseMimeType: 'application/json',
                    responseJsonSchema: CatalogSchema,
                }
            });

            const responseText = fallbackResponse.text || '{}';
            const catalogData = JSON.parse(responseText);

            return {
                products: catalogData.products,
                categories: catalogData.categories,
                totalValue: catalogData.products.reduce((sum: number, p: Product) => sum + p.price, 0),
                generatedAt: new Date(),
                context: businessContext
            };
        } catch (fallbackError) {
            console.error('Fallback model also failed:', fallbackError);

            // Ultimate fallback: Return static demo catalog when quota exhausted
            console.warn('⚠️ Using static demo catalog due to API quota limits');
            return getStaticDemoCatalog(businessContext);
        }
    }
}

// === STATIC DEMO CATALOG (Fallback for rate limits) ===
function getStaticDemoCatalog(businessContext: string): ProductCatalog {
    const products: Product[] = [
        {
            id: 'PROD-001',
            name: 'Premium Wireless Headphones',
            category: 'Electronics',
            price: 2499,
            keywords: ['wireless', 'bluetooth', 'audio', 'noise-cancelling'],
            description: 'High-quality wireless headphones with active noise cancellation',
            targetAudience: ['professionals', 'students', 'music lovers'],
            seasonality: 'ALL_YEAR'
        },
        {
            id: 'PROD-002',
            name: 'Laptop Stand Aluminum',
            category: 'Accessories',
            price: 899,
            keywords: ['laptop', 'ergonomic', 'desk', 'aluminum'],
            description: 'Ergonomic laptop stand for better posture',
            targetAudience: ['remote workers', 'students'],
            seasonality: 'ALL_YEAR'
        },
        {
            id: 'PROD-003',
            name: 'USB-C Hub Multi-Port',
            category: 'Electronics',
            price: 1299,
            keywords: ['usb-c', 'hub', 'adapter', 'ports'],
            description: '7-in-1 USB-C hub with HDMI, USB 3.0, and card readers',
            targetAudience: ['professionals', 'digital nomads'],
            seasonality: 'ALL_YEAR'
        },
        {
            id: 'PROD-004',
            name: 'Mechanical Keyboard RGB',
            category: 'Electronics',
            price: 3499,
            keywords: ['keyboard', 'mechanical', 'rgb', 'gaming'],
            description: 'Mechanical gaming keyboard with RGB backlighting',
            targetAudience: ['gamers', 'programmers'],
            seasonality: 'ALL_YEAR'
        },
        {
            id: 'PROD-005',
            name: 'Portable SSD 1TB',
            category: 'Storage',
            price: 4999,
            keywords: ['ssd', 'storage', 'portable', 'backup'],
            description: 'Fast portable SSD with 1TB capacity',
            targetAudience: ['content creators', 'professionals'],
            seasonality: 'ALL_YEAR'
        },
        {
            id: 'PROD-006',
            name: 'Webcam 1080p HD',
            category: 'Electronics',
            price: 1899,
            keywords: ['webcam', 'video', 'streaming', 'meetings'],
            description: 'Full HD webcam for video calls and streaming',
            targetAudience: ['remote workers', 'streamers'],
            seasonality: 'ALL_YEAR'
        },
        {
            id: 'PROD-007',
            name: 'Phone Stand Adjustable',
            category: 'Accessories',
            price: 399,
            keywords: ['phone', 'stand', 'holder', 'desk'],
            description: 'Adjustable phone stand for desk or nightstand',
            targetAudience: ['everyone'],
            seasonality: 'ALL_YEAR'
        },
        {
            id: 'PROD-008',
            name: 'Wireless Mouse Ergonomic',
            category: 'Electronics',
            price: 799,
            keywords: ['mouse', 'wireless', 'ergonomic', 'bluetooth'],
            description: 'Ergonomic wireless mouse with precision tracking',
            targetAudience: ['professionals', 'students'],
            seasonality: 'ALL_YEAR'
        },
        {
            id: 'PROD-009',
            name: 'Cable Management Kit',
            category: 'Accessories',
            price: 299,
            keywords: ['cables', 'organizer', 'desk', 'management'],
            description: 'Complete cable management solution for clean desk setup',
            targetAudience: ['everyone'],
            seasonality: 'ALL_YEAR'
        },
        {
            id: 'PROD-010',
            name: 'Desk Lamp LED Smart',
            category: 'Accessories',
            price: 1599,
            keywords: ['lamp', 'led', 'desk', 'smart', 'adjustable'],
            description: 'Smart LED desk lamp with adjustable brightness and color',
            targetAudience: ['students', 'professionals'],
            seasonality: 'ALL_YEAR'
        }
    ];

    return {
        products,
        categories: ['Electronics', 'Accessories', 'Storage'],
        totalValue: products.reduce((sum, p) => sum + p.price, 0),
        generatedAt: new Date(),
        context: `${businessContext} (demo mode - API quota reached)`
    };
}

// === SMART RECOMMENDATIONS ===

/**
 * Generate intelligent product recommendations using Gemini AI
 * PRIMARY recommendation engine - uses deep context understanding
 */
export async function generateSmartRecommendations(
    request: RecommendationRequest,
    catalog: Product[]
): Promise<RecommendationResult> {
    const client = getGeminiClient();

    const targetProduct = catalog.find(p => p.id === request.productId);
    if (!targetProduct) {
        throw new Error(`Product ${request.productId} not found in catalog`);
    }

    const prompt = buildRecommendationPrompt(targetProduct, catalog, request);

    try {
        // Note: codeExecution cannot be used with structured JSON output
        const response = await client.models.generateContent({
            model: PRIMARY_MODEL,
            contents: prompt,
            config: {
                responseMimeType: 'application/json',
                responseJsonSchema: RecommendationSchema,
            }
        });

        const responseText = response.text || '{}';
        const result = JSON.parse(responseText);

        // Map product IDs back to full product objects
        const recommendations: GeminiRecommendation[] = result.recommendations.map((rec: any) => {
            const product = catalog.find(p => p.id === rec.productId);
            if (!product) {
                throw new Error(`Recommended product ${rec.productId} not found`);
            }
            return {
                product,
                score: rec.score,
                reason: rec.reason,
                insights: rec.insights,
                bundleCompatibility: rec.bundleCompatibility
            };
        });

        // RECALCULATE BUNDLE MATH (ensure correctness via TypeScript)
        if (result.analysis && result.analysis.bundleOpportunities) {
            result.analysis.bundleOpportunities = result.analysis.bundleOpportunities.map((bundle: any) => {
                let totalListPrice = 0;
                bundle.products.forEach((prodIdOrName: string) => {
                    // Try to find by ID or Name (AI might return either)
                    const p = catalog.find(cp => cp.id === prodIdOrName || cp.name === prodIdOrName);
                    if (p) totalListPrice += p.price;
                });

                // Recalculate expected value if we found the products
                if (totalListPrice > 0) {
                    const discountFactor = 1 - (bundle.discount / 100);
                    bundle.expectedValue = Math.round(totalListPrice * discountFactor);
                }
                return bundle;
            });
        }

        return {
            recommendations,
            strategy: result.strategy,
            confidence: result.confidence,
            analysis: result.analysis,
            tensorFlowEnhanced: false
        };

    } catch (error) {
        console.error('Gemini recommendation error:', error);
        throw new Error(`Failed to generate recommendations: ${error}`);
    }
}

/**
 * Enhance Gemini recommendations with TensorFlow similarity scores (optional)
 */
export async function enhanceWithTensorFlow(
    geminiResult: RecommendationResult,
    calculateSimilarity: (a: Product, b: Product, catalog: Product[]) => Promise<number>,
    targetProduct: Product,
    catalog: Product[]
): Promise<RecommendationResult> {
    try {
        // Calculate TF similarity scores for Gemini's recommendations
        const enhanced = await Promise.all(
            geminiResult.recommendations.map(async (rec) => {
                const tfScore = await calculateSimilarity(targetProduct, rec.product, catalog);

                // Blend Gemini intelligence with TF similarity (70% Gemini, 30% TF)
                const blendedScore = (rec.score * 0.7) + (tfScore * 100 * 0.3);

                return {
                    ...rec,
                    score: Math.round(blendedScore),
                    insights: [
                        ...rec.insights,
                        `TensorFlow similarity: ${(tfScore * 100).toFixed(1)}%`
                    ]
                };
            })
        );

        return {
            ...geminiResult,
            recommendations: enhanced,
            tensorFlowEnhanced: true
        };
    } catch (error) {
        console.error('TensorFlow enhancement failed:', error);
        // Return original Gemini result if TF fails
        return geminiResult;
    }
}

// === HELPER FUNCTIONS ===

function buildRecommendationPrompt(
    targetProduct: Product,
    catalog: Product[],
    request: RecommendationRequest
): string {
    const strategy = request.strategy || 'smart';
    const customerInfo = request.customerProfile
        ? `\nCustomer Profile:
- Budget: ${request.customerProfile.budget || 'Not specified'}
- Preferences: ${request.customerProfile.preferences?.join(', ') || 'None'}
- Purchase History: ${request.customerProfile.purchaseHistory?.join(', ') || 'None'}`
        : '';

    return `You are an expert e-commerce recommendation engine.

TARGET PRODUCT:
Name: ${targetProduct.name}
Category: ${targetProduct.category}
Price: ₹${targetProduct.price}
Description: ${targetProduct.description}
Keywords: ${targetProduct.keywords.join(', ')}

AVAILABLE CATALOG:
${catalog.map(p => `- [${p.id}] ${p.name} (${p.category}) - ₹${p.price}`).join('\n')}

STRATEGY: ${strategy}
${customerInfo}
${request.context ? `\nADDITIONAL CONTEXT: ${request.context}` : ''}

TASK:
Generate top product recommendations.
Make up a reason, insights, and bundle opportunities.
Use standard strategic analysis.

Return structured JSON following the schema.`;
}

/**
 * Cache for generated catalogs (in-memory for now, can move to Redis)
 */
let cachedCatalog: ProductCatalog | null = null;
let cacheExpiry: Date | null = null;
const CACHE_TTL_MS = 3600000; // 1 hour

export async function getCachedCatalog(
    businessContext: string = 'electronics and accessories e-commerce',
    forceRefresh: boolean = false
): Promise<ProductCatalog> {
    const now = new Date();

    if (!forceRefresh && cachedCatalog && cacheExpiry && now < cacheExpiry) {
        return cachedCatalog;
    }

    // Generate fresh catalog
    cachedCatalog = await generateProductCatalog(businessContext);
    cacheExpiry = new Date(now.getTime() + CACHE_TTL_MS);

    return cachedCatalog;
}

export function clearCatalogCache(): void {
    cachedCatalog = null;
    cacheExpiry = null;
}
