/**
 * Gemini AI Product Catalog Generator
 * Dynamically generates product catalogs based on business context
 * PRIMARY AI engine for SellerOps product intelligence
 */

import { GoogleGenAI } from '@google/genai';

// === MODELS ===
const PRIMARY_MODEL = 'gemini-2.5-flash';
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
        specifications: { type: 'object' },
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

Generate ${productCount} realistic, diverse products with the following requirements:

1. Product diversity: Mix of different price points, categories, and use cases
2. Realistic pricing: Use appropriate currency formatting (e.g., ₹ for INR)
3. Rich metadata: Include keywords, specifications, and target audience
4. Seasonality: Mark products as ALL_YEAR, SEASONAL, or TRENDING
5. Complementary items: Some products should naturally complement others

Generate products that would actually sell together and make business sense.
Include both high-margin and volume products.

Return the catalog as structured JSON matching the schema.`;

    const config: any = {
        responseMimeType: 'application/json',
        responseSchema: CatalogSchema,
    };

    if (useCodeExecution) {
        config.tools = [{ codeExecution: {} }];
    }

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
                    responseSchema: CatalogSchema,
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
            throw new Error(`Failed to generate catalog: ${fallbackError}`);
        }
    }
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
        const response = await client.models.generateContent({
            model: PRIMARY_MODEL,
            contents: prompt,
            config: {
                responseMimeType: 'application/json',
                responseSchema: RecommendationSchema,
                tools: [{ codeExecution: {} }]
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
    calculateSimilarity: (a: Product, b: Product) => Promise<number>,
    targetProduct: Product
): Promise<RecommendationResult> {
    try {
        // Calculate TF similarity scores for Gemini's recommendations
        const enhanced = await Promise.all(
            geminiResult.recommendations.map(async (rec) => {
                const tfScore = await calculateSimilarity(targetProduct, rec.product);
                
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
Generate the top 3-5 product recommendations using ${strategy} strategy.

For each recommendation:
1. Provide a clear reason WHY this product pairs well
2. Generate actionable insights (e.g., "Bundle these for 15% off")
3. Rate bundle compatibility (0-100)

Also provide:
- Strategic analysis with cross-sell/upsell/bundle insights
- Bundle opportunities with specific discounts
- Expected business impact (revenue increase, conversion boost)
- Pricing strategy recommendations

Use Python code execution if needed to calculate optimal bundle prices or analyze patterns.

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
