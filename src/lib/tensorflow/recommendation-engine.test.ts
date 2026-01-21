/**
 * Tests for TensorFlow Recommendation Engine
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as tf from '@tensorflow/tfjs';
import {
    calculateSimilarity,
    getSimilarProducts,
    getSmartRecommendations,
    getTensorFlowInfo,
    DEMO_PRODUCTS,
    generateSimilarityMatrix
} from './recommendation-engine';

describe('TensorFlow Recommendation Engine', () => {
    beforeAll(async () => {
        // Ensure TensorFlow is ready
        await tf.ready();
    });

    describe('TensorFlow Backend', () => {
        it('should have TensorFlow initialized', () => {
            const info = getTensorFlowInfo();
            expect(info.ready).toBe(true);
            expect(info.backend).toBeTruthy();
        });

        it('should use CPU or WebGL backend', () => {
            const backend = tf.getBackend();
            expect(['cpu', 'webgl', 'wasm']).toContain(backend);
        });
    });

    describe('Product Similarity Calculation', () => {
        it('should calculate similarity between two products', async () => {
            const productA = DEMO_PRODUCTS[0]; // Wireless Earbuds
            const productB = DEMO_PRODUCTS[5]; // Bluetooth Speaker

            const similarity = await calculateSimilarity(productA, productB);

            expect(similarity).toBeGreaterThanOrEqual(0);
            expect(similarity).toBeLessThanOrEqual(1);
            expect(typeof similarity).toBe('number');
        });

        it('should give high similarity for same category products', async () => {
            const earbuds = DEMO_PRODUCTS[0]; // Electronics
            const speaker = DEMO_PRODUCTS[5]; // Electronics

            const similarity = await calculateSimilarity(earbuds, speaker);

            // Same category + similar keywords should have decent similarity
            expect(similarity).toBeGreaterThan(0.3);
        });

        it('should give lower similarity for different categories', async () => {
            const earbuds = DEMO_PRODUCTS[0]; // Electronics
            const phoneCase = DEMO_PRODUCTS[1]; // Accessories

            const similarity = await calculateSimilarity(earbuds, phoneCase);

            // Different categories should have lower similarity
            expect(similarity).toBeLessThan(0.8);
        });

        it('should return 1.0 for identical products', async () => {
            const product = DEMO_PRODUCTS[0];
            const similarity = await calculateSimilarity(product, product);

            // Same product should have perfect similarity
            expect(similarity).toBeCloseTo(1.0, 1);
        });
    });

    describe('Similar Products Recommendation', () => {
        it('should return top K similar products', async () => {
            const productId = 'prod-1'; // Wireless Earbuds
            const topK = 3;

            const recommendations = await getSimilarProducts(productId, topK);

            expect(recommendations).toHaveLength(topK);
            expect(recommendations[0].product.id).not.toBe(productId);
        });

        it('should sort recommendations by score descending', async () => {
            const recommendations = await getSimilarProducts('prod-1', 5);

            for (let i = 0; i < recommendations.length - 1; i++) {
                expect(recommendations[i].score).toBeGreaterThanOrEqual(
                    recommendations[i + 1].score
                );
            }
        });

        it('should include reason for recommendation', async () => {
            const recommendations = await getSimilarProducts('prod-1', 2);

            recommendations.forEach(rec => {
                expect(rec.reason).toBeTruthy();
                expect(typeof rec.reason).toBe('string');
            });
        });

        it('should throw error for non-existent product', async () => {
            await expect(
                getSimilarProducts('non-existent-id')
            ).rejects.toThrow();
        });
    });

    describe('Smart Recommendations', () => {
        it('should generate mixed strategy recommendations', async () => {
            const result = await getSmartRecommendations('prod-1', 'mixed');

            expect(result.recommendations.length).toBeGreaterThan(0);
            expect(result.strategy).toBe('mixed');
            expect(result.confidence).toBeGreaterThan(0);
            expect(result.confidence).toBeLessThanOrEqual(100);
        });

        it('should filter out low-score recommendations', async () => {
            const result = await getSmartRecommendations('prod-1', 'mixed');

            result.recommendations.forEach(rec => {
                expect(rec.score).toBeGreaterThan(30);
            });
        });

        it('should support similar strategy', async () => {
            const result = await getSmartRecommendations('prod-1', 'similar');

            expect(result.strategy).toBe('similar');
            expect(result.recommendations.length).toBeGreaterThan(0);
        });

        it('should support complementary strategy', async () => {
            const result = await getSmartRecommendations('prod-1', 'complementary');

            expect(result.strategy).toBe('complementary');
        });

        it('should support upsell strategy', async () => {
            const result = await getSmartRecommendations('prod-1', 'upsell');

            expect(result.strategy).toBe('upsell');
        });
    });

    describe('Similarity Matrix Generation', () => {
        it('should generate full similarity matrix', async () => {
            const matrix = await generateSimilarityMatrix(DEMO_PRODUCTS);

            expect(matrix.length).toBe(DEMO_PRODUCTS.length);
            expect(matrix[0].length).toBe(DEMO_PRODUCTS.length);
        });

        it('should have diagonal values close to 1', async () => {
            const matrix = await generateSimilarityMatrix(DEMO_PRODUCTS.slice(0, 3));

            for (let i = 0; i < matrix.length; i++) {
                expect(matrix[i][i]).toBeCloseTo(1.0, 1);
            }
        });

        it('should be symmetric', async () => {
            const matrix = await generateSimilarityMatrix(DEMO_PRODUCTS.slice(0, 3));

            for (let i = 0; i < matrix.length; i++) {
                for (let j = i + 1; j < matrix.length; j++) {
                    expect(matrix[i][j]).toBeCloseTo(matrix[j][i], 2);
                }
            }
        });
    });

    describe('Demo Product Catalog', () => {
        it('should have valid demo products', () => {
            expect(DEMO_PRODUCTS.length).toBeGreaterThan(0);

            DEMO_PRODUCTS.forEach(product => {
                expect(product.id).toBeTruthy();
                expect(product.name).toBeTruthy();
                expect(product.category).toBeTruthy();
                expect(product.price).toBeGreaterThan(0);
                expect(product.keywords.length).toBeGreaterThan(0);
                expect(product.description).toBeTruthy();
            });
        });

        it('should have multiple categories', () => {
            const categories = new Set(DEMO_PRODUCTS.map(p => p.category));
            expect(categories.size).toBeGreaterThan(1);
        });

        it('should have price variation', () => {
            const prices = DEMO_PRODUCTS.map(p => p.price);
            const minPrice = Math.min(...prices);
            const maxPrice = Math.max(...prices);

            expect(maxPrice).toBeGreaterThan(minPrice);
        });
    });

    describe('Memory Management', () => {
        it('should not leak tensors', async () => {
            const initialTensors = tf.memory().numTensors;

            // Perform multiple operations
            await getSimilarProducts('prod-1', 3);
            await calculateSimilarity(DEMO_PRODUCTS[0], DEMO_PRODUCTS[1]);

            const finalTensors = tf.memory().numTensors;

            // Allow tolerance for TensorFlow internal caching and Node.js backend
            expect(finalTensors - initialTensors).toBeLessThan(30);
        });
    });
});
