/**
 * TensorFlow.js Smart Recommendation Engine
 * Uses product embeddings and similarity for zero-data recommendations
 * No historical data needed - works with product features
 */

import * as tf from '@tensorflow/tfjs';

export interface Product {
    id: string;
    name: string;
    category: string;
    price: number;
    keywords: string[];
    description: string;
}

export interface ProductRecommendation {
    product: Product;
    score: number;
    reason: string;
}

export interface RecommendationResult {
    recommendations: ProductRecommendation[];
    strategy: string;
    confidence: number;
}

// === DEMO PRODUCT CATALOG ===
export const DEMO_PRODUCTS: Product[] = [
    {
        id: 'prod-1',
        name: 'Wireless Bluetooth Earbuds',
        category: 'Electronics',
        price: 2499,
        keywords: ['audio', 'wireless', 'bluetooth', 'music', 'portable'],
        description: 'Premium wireless earbuds with noise cancellation'
    },
    {
        id: 'prod-2',
        name: 'Phone Case - Silicone',
        category: 'Accessories',
        price: 299,
        keywords: ['protection', 'phone', 'silicone', 'durable', 'accessories'],
        description: 'Durable silicone phone case with drop protection'
    },
    {
        id: 'prod-3',
        name: 'USB-C Fast Charger',
        category: 'Electronics',
        price: 899,
        keywords: ['charging', 'usb-c', 'fast', 'power', 'electronics'],
        description: '30W USB-C fast charging adapter'
    },
    {
        id: 'prod-4',
        name: 'Portable Power Bank 10000mAh',
        category: 'Electronics',
        price: 1299,
        keywords: ['battery', 'portable', 'charging', 'power', 'mobile'],
        description: 'High-capacity portable power bank with dual USB ports'
    },
    {
        id: 'prod-5',
        name: 'Screen Protector - Tempered Glass',
        category: 'Accessories',
        price: 199,
        keywords: ['protection', 'screen', 'glass', 'phone', 'accessories'],
        description: '9H hardness tempered glass screen protector'
    },
    {
        id: 'prod-6',
        name: 'Bluetooth Speaker - Waterproof',
        category: 'Electronics',
        price: 1899,
        keywords: ['audio', 'speaker', 'bluetooth', 'waterproof', 'music'],
        description: 'Portable waterproof Bluetooth speaker with 12hr battery'
    },
    {
        id: 'prod-7',
        name: 'Laptop Sleeve - 15 inch',
        category: 'Accessories',
        price: 599,
        keywords: ['laptop', 'protection', 'sleeve', 'portable', 'accessories'],
        description: 'Padded laptop sleeve with extra pocket'
    },
    {
        id: 'prod-8',
        name: 'Wireless Mouse - Ergonomic',
        category: 'Electronics',
        price: 799,
        keywords: ['mouse', 'wireless', 'ergonomic', 'computer', 'productivity'],
        description: 'Ergonomic wireless mouse with adjustable DPI'
    },
    {
        id: 'prod-9',
        name: 'HDMI Cable 2m',
        category: 'Accessories',
        price: 249,
        keywords: ['cable', 'hdmi', 'video', 'display', 'connectivity'],
        description: '4K HDMI 2.1 cable for high-speed video transmission'
    },
    {
        id: 'prod-10',
        name: 'Smart Watch Strap',
        category: 'Accessories',
        price: 399,
        keywords: ['watch', 'strap', 'wearable', 'fashion', 'accessories'],
        description: 'Premium silicone strap for smart watches'
    }
];

// === FEATURE EXTRACTION ===

/**
 * Convert product to numerical feature vector for TensorFlow
 * Features: category, price tier, keyword overlap
 */
function productToFeatureVector(product: Product, allProducts: Product[]): number[] {
    // Category encoding (one-hot)
    const categories = ['Electronics', 'Accessories', 'Other'];
    const categoryFeatures = categories.map(cat => product.category === cat ? 1 : 0);

    // Price tier (normalized 0-1)
    const prices = allProducts.map(p => p.price);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const normalizedPrice = (product.price - minPrice) / (maxPrice - minPrice || 1);

    // Keyword features (bag of words approach)
    const allKeywords = new Set<string>();
    allProducts.forEach(p => p.keywords.forEach(k => allKeywords.add(k)));
    const keywordVector = Array.from(allKeywords).map(keyword =>
        product.keywords.includes(keyword) ? 1 : 0
    );

    return [...categoryFeatures, normalizedPrice, ...keywordVector];
}

// === SIMILARITY COMPUTATION ===

/**
 * Calculate cosine similarity between two products using TensorFlow
 */
export async function calculateSimilarity(
    productA: Product,
    productB: Product,
    allProducts: Product[] = DEMO_PRODUCTS
): Promise<number> {
    const vectorA = productToFeatureVector(productA, allProducts);
    const vectorB = productToFeatureVector(productB, allProducts);

    const tensorA = tf.tensor1d(vectorA);
    const tensorB = tf.tensor1d(vectorB);

    // Cosine similarity: (A · B) / (||A|| * ||B||)
    const dotProduct = tf.sum(tf.mul(tensorA, tensorB));
    const normA = tf.norm(tensorA);
    const normB = tf.norm(tensorB);

    const similarity = tf.div(dotProduct, tf.mul(normA, normB));
    const score = await similarity.data();

    // Cleanup tensors
    tensorA.dispose();
    tensorB.dispose();
    dotProduct.dispose();
    normA.dispose();
    normB.dispose();
    similarity.dispose();

    return score[0];
}

// === RECOMMENDATION STRATEGIES ===

/**
 * Get similar products based on TensorFlow cosine similarity
 */
export async function getSimilarProducts(
    productId: string,
    topK: number = 3,
    catalog: Product[] = DEMO_PRODUCTS
): Promise<ProductRecommendation[]> {
    const targetProduct = catalog.find(p => p.id === productId);
    if (!targetProduct) {
        throw new Error(`Product ${productId} not found`);
    }

    // Calculate similarities for all other products
    const similarities = await Promise.all(
        catalog
            .filter(p => p.id !== productId)
            .map(async (product) => ({
                product,
                score: await calculateSimilarity(targetProduct, product, catalog)
            }))
    );

    // Sort by similarity and take top K
    const recommendations = similarities
        .sort((a, b) => b.score - a.score)
        .slice(0, topK)
        .map(({ product, score }) => ({
            product,
            score: Math.round(score * 100),
            reason: generateReason(targetProduct, product, score)
        }));

    return recommendations;
}

/**
 * Get complementary products (cross-sell strategy)
 */
export async function getComplementaryProducts(
    productId: string,
    topK: number = 3,
    catalog: Product[] = DEMO_PRODUCTS
): Promise<ProductRecommendation[]> {
    const targetProduct = catalog.find(p => p.id === productId);
    if (!targetProduct) {
        throw new Error(`Product ${productId} not found`);
    }

    // Find products in different category but related keywords
    const complementary = await Promise.all(
        catalog
            .filter(p => p.id !== productId && p.category !== targetProduct.category)
            .map(async (product) => {
                const keywordOverlap = targetProduct.keywords.filter(k =>
                    product.keywords.includes(k)
                ).length;

                const similarity = await calculateSimilarity(targetProduct, product, catalog);
                const complementaryScore = keywordOverlap * 0.4 + similarity * 0.6;

                return {
                    product,
                    score: complementaryScore
                };
            })
    );

    return complementary
        .sort((a, b) => b.score - a.score)
        .slice(0, topK)
        .map(({ product, score }) => ({
            product,
            score: Math.round(score * 100),
            reason: `Commonly bought together with ${targetProduct.name}`
        }));
}

/**
 * Get upsell products (higher price, same category)
 */
export async function getUpsellProducts(
    productId: string,
    topK: number = 2,
    catalog: Product[] = DEMO_PRODUCTS
): Promise<ProductRecommendation[]> {
    const targetProduct = catalog.find(p => p.id === productId);
    if (!targetProduct) {
        throw new Error(`Product ${productId} not found`);
    }

    const upsells = await Promise.all(
        catalog
            .filter(p =>
                p.id !== productId &&
                p.category === targetProduct.category &&
                p.price > targetProduct.price
            )
            .map(async (product) => ({
                product,
                score: await calculateSimilarity(targetProduct, product, catalog)
            }))
    );

    return upsells
        .sort((a, b) => b.score - a.score)
        .slice(0, topK)
        .map(({ product, score }) => ({
            product,
            score: Math.round(score * 100),
            reason: `Premium alternative with ${Math.round(((product.price - targetProduct.price) / targetProduct.price) * 100)}% higher value`
        }));
}

// === SMART RECOMMENDATION ENGINE ===

/**
 * Main recommendation function - combines multiple strategies
 */
export async function getSmartRecommendations(
    productId: string,
    strategy: 'similar' | 'complementary' | 'upsell' | 'mixed' = 'mixed',
    catalog: Product[] = DEMO_PRODUCTS
): Promise<RecommendationResult> {
    let recommendations: ProductRecommendation[] = [];
    let strategyUsed = strategy;
    let confidence = 85;

    switch (strategy) {
        case 'similar':
            recommendations = await getSimilarProducts(productId, 3, catalog);
            confidence = 90;
            break;

        case 'complementary':
            recommendations = await getComplementaryProducts(productId, 3, catalog);
            confidence = 82;
            break;

        case 'upsell':
            recommendations = await getUpsellProducts(productId, 2, catalog);
            confidence = 75;
            break;

        case 'mixed':
        default:
            // Intelligent mix: 1 similar + 1 complementary + 1 upsell
            const [similar, complementary, upsell] = await Promise.all([
                getSimilarProducts(productId, 1, catalog),
                getComplementaryProducts(productId, 1, catalog),
                getUpsellProducts(productId, 1, catalog)
            ]);
            recommendations = [...similar, ...complementary, ...upsell];
            strategyUsed = 'mixed';
            confidence = 85;
            break;
    }

    return {
        recommendations: recommendations.filter(r => r.score > 30), // Filter low scores
        strategy: strategyUsed,
        confidence
    };
}

// === HELPER FUNCTIONS ===

function generateReason(productA: Product, productB: Product, similarity: number): string {
    if (similarity > 0.8) {
        return `Very similar to ${productA.name}`;
    } else if (similarity > 0.6) {
        return `Customers also viewed this`;
    } else if (similarity > 0.4) {
        return `Related product in ${productB.category}`;
    }
    return 'Recommended for you';
}

// === BATCH SIMILARITY MATRIX (for advanced use) ===

/**
 * Generate similarity matrix for entire catalog using TensorFlow batch operations
 */
export async function generateSimilarityMatrix(
    catalog: Product[] = DEMO_PRODUCTS
): Promise<number[][]> {
    const featureVectors = catalog.map(p => productToFeatureVector(p, catalog));
    const matrix = tf.tensor2d(featureVectors);

    // Compute all pairwise similarities efficiently
    const matrixT = tf.transpose(matrix);
    const dotProducts = tf.matMul(matrix, matrixT);

    const norms = tf.norm(matrix, 'euclidean', 1, true);
    const normsT = tf.transpose(norms);
    const normProducts = tf.matMul(norms, normsT);

    const similarityMatrix = tf.div(dotProducts, normProducts);
    const result = await similarityMatrix.array() as number[][];

    // Cleanup
    matrix.dispose();
    matrixT.dispose();
    dotProducts.dispose();
    norms.dispose();
    normsT.dispose();
    normProducts.dispose();
    similarityMatrix.dispose();

    return result;
}

/**
 * Get TensorFlow backend info (for debugging/UI display)
 */
export function getTensorFlowInfo(): { backend: string; ready: boolean } {
    return {
        backend: tf.getBackend(),
        ready: tf.engine().backendInstance !== null
    };
}
