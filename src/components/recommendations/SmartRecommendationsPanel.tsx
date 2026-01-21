/**
 * Smart Recommendations Panel Component
 * Showcases TensorFlow + Gemini AI recommendation engine
 */

'use client';

import { useState } from 'react';
import type { Product } from '@/lib/tensorflow/recommendation-engine';

interface RecommendationResponse {
    targetProduct: Product;
    recommendations: Array<{
        product: Product;
        score: number;
        reason: string;
    }>;
    strategy: string;
    tfConfidence: number;
    analysis: {
        summary: string;
        insights: Array<{
            type: string;
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
        confidence: number;
    };
    powered: {
        tensorflow: boolean;
        gemini: boolean;
        backend: string;
    };
}

interface ProductCatalog {
    products: Product[];
    tensorFlowBackend: string;
    tensorFlowReady: boolean;
}

export function SmartRecommendationsPanel() {
    const [catalog, setCatalog] = useState<ProductCatalog | null>(null);
    const [selectedProductId, setSelectedProductId] = useState<string>('');
    const [result, setResult] = useState<RecommendationResponse | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [useGemini, setUseGemini] = useState(false);
    const [mounted, setMounted] = useState(false);

    // Client-side only initialization
    if (typeof window !== 'undefined' && !mounted) {
        setMounted(true);
    }

    // Load product catalog
    const loadCatalog = async () => {
        if (typeof window === 'undefined') return; // Skip on server
        
        try {
            const response = await fetch('/api/recommendations');
            if (!response.ok) throw new Error('Failed to fetch');
            const data = await response.json();
            setCatalog(data);
            if (data.products.length > 0) {
                setSelectedProductId(data.products[0].id);
            }
        } catch (error) {
            console.error('Failed to load catalog:', error);
        }
    };

    // Generate recommendations
    const generateRecommendations = async () => {
        if (!selectedProductId) return;

        setIsLoading(true);
        try {
            const response = await fetch('/api/recommendations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    productId: selectedProductId,
                    strategy: 'mixed',
                    useGeminiAnalysis: useGemini
                })
            });

            if (!response.ok) throw new Error('Failed to generate recommendations');

            const data = await response.json();
            setResult(data);
        } catch (error) {
            console.error('Recommendation error:', error);
        } finally {
            setIsLoading(false);
        }
    };

    // Initial load - client-side only
    if (!catalog && mounted) {
        loadCatalog();
    }

    if (!mounted || !catalog) {
        return (
            <div className="flex-1 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-pulse font-mono text-signal-cyan">
                        ⚡ Initializing TensorFlow.js...
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="flex-1 flex flex-col overflow-hidden">
            {/* Header */}
            <div className="hud-panel p-6 border-b border-border-default">
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h2 className="font-mono text-xl font-bold text-signal-cyan mb-1">
                            🤖 SMART RECOMMENDATIONS
                        </h2>
                        <p className="text-sm text-text-secondary">
                            TensorFlow.js Product Similarity + Gemini AI Analysis
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        <div className="font-mono text-xs text-signal-green">
                            ✓ TF Backend: {catalog.tensorFlowBackend}
                        </div>
                        <div className="font-mono text-xs text-signal-amber">
                            {useGemini ? '✓ Gemini AI: Active' : '○ Gemini AI: Off'}
                        </div>
                    </div>
                </div>

                {/* Controls */}
                <div className="flex gap-4">
                    <select
                        value={selectedProductId}
                        onChange={(e) => setSelectedProductId(e.target.value)}
                        className="flex-1 bg-surface-elevated border border-border-default px-4 py-2 font-mono text-sm text-text-primary"
                    >
                        {catalog.products.map(product => (
                            <option key={product.id} value={product.id}>
                                {product.name} - ₹{product.price}
                            </option>
                        ))}
                    </select>

                    <label className="flex items-center gap-2 px-4 bg-surface-elevated border border-border-default cursor-pointer hover:border-signal-amber transition-colors">
                        <input
                            type="checkbox"
                            checked={useGemini}
                            onChange={(e) => setUseGemini(e.target.checked)}
                            className="w-4 h-4"
                        />
                        <span className="font-mono text-xs uppercase tracking-wider text-text-secondary">
                            Use Gemini Analysis
                        </span>
                    </label>

                    <button
                        onClick={generateRecommendations}
                        disabled={isLoading}
                        className="px-8 py-2 bg-signal-green text-surface-base font-mono text-sm font-bold uppercase tracking-wider hover:bg-opacity-90 transition-opacity disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {isLoading ? 'Analyzing...' : 'Generate'}
                    </button>
                </div>
            </div>

            {/* Results */}
            {result && (
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                    {/* Target Product */}
                    <div className="hud-panel p-4 border-l-4 border-signal-cyan">
                        <div className="font-mono text-xs uppercase tracking-wider text-text-tertiary mb-2">
                            Target Product
                        </div>
                        <h3 className="font-mono text-lg font-bold text-text-primary mb-1">
                            {result.targetProduct.name}
                        </h3>
                        <div className="flex items-center gap-4 text-sm text-text-secondary">
                            <span>{result.targetProduct.category}</span>
                            <span>₹{result.targetProduct.price}</span>
                            <span className="text-signal-cyan">
                                Keywords: {result.targetProduct.keywords.join(', ')}
                            </span>
                        </div>
                    </div>

                    {/* AI Analysis Summary */}
                    <div className="hud-panel p-6 border-l-4 border-signal-amber">
                        <div className="flex items-center justify-between mb-4">
                            <div className="font-mono text-sm uppercase tracking-wider text-signal-amber">
                                🧠 AI Analysis
                            </div>
                            <div className="flex gap-2 text-xs font-mono">
                                <span className="text-text-tertiary">TF Confidence:</span>
                                <span className="text-signal-green font-bold">{result.tfConfidence}%</span>
                                <span className="text-text-tertiary ml-3">Gemini Confidence:</span>
                                <span className="text-signal-green font-bold">{result.analysis.confidence}%</span>
                            </div>
                        </div>
                        <p className="text-text-primary leading-relaxed mb-4">
                            {result.analysis.summary}
                        </p>
                        <div className="grid grid-cols-2 gap-4 pt-4 border-t border-border-default">
                            <div>
                                <div className="text-xs text-text-tertiary mb-1">Expected Revenue Increase</div>
                                <div className="text-xl font-bold text-signal-green">
                                    {result.analysis.expectedImpact.revenueIncrease}
                                </div>
                            </div>
                            <div>
                                <div className="text-xs text-text-tertiary mb-1">Conversion Boost</div>
                                <div className="text-xl font-bold text-signal-green">
                                    {result.analysis.expectedImpact.conversionBoost}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Recommendations */}
                    <div className="space-y-3">
                        <h3 className="font-mono text-sm uppercase tracking-wider text-text-tertiary">
                            Recommended Products ({result.recommendations.length})
                        </h3>
                        {result.recommendations.map((rec, idx) => (
                            <div key={idx} className="hud-panel p-4 hover:border-signal-cyan transition-colors">
                                <div className="flex items-start justify-between mb-2">
                                    <div className="flex-1">
                                        <h4 className="font-mono font-bold text-text-primary mb-1">
                                            {rec.product.name}
                                        </h4>
                                        <p className="text-sm text-text-secondary mb-2">
                                            {rec.reason}
                                        </p>
                                        <div className="flex items-center gap-3 text-xs text-text-tertiary">
                                            <span>{rec.product.category}</span>
                                            <span className="text-text-primary font-bold">₹{rec.product.price}</span>
                                        </div>
                                    </div>
                                    <div className="ml-4 text-center">
                                        <div className="text-xs text-text-tertiary mb-1">Match Score</div>
                                        <div className={`text-2xl font-bold font-mono ${
                                            rec.score > 80 ? 'text-signal-green' :
                                            rec.score > 60 ? 'text-signal-amber' :
                                            'text-signal-cyan'
                                        }`}>
                                            {rec.score}%
                                        </div>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Insights */}
                    {result.analysis.insights.length > 0 && (
                        <div className="space-y-3">
                            <h3 className="font-mono text-sm uppercase tracking-wider text-text-tertiary">
                                Strategic Insights
                            </h3>
                            {result.analysis.insights.map((insight, idx) => (
                                <div key={idx} className="hud-panel p-4 border-l-4 border-signal-green">
                                    <div className="flex items-start justify-between">
                                        <div className="flex-1">
                                            <div className="font-mono text-xs uppercase tracking-wider text-signal-green mb-1">
                                                {insight.type}
                                            </div>
                                            <h4 className="font-mono font-bold text-text-primary mb-1">
                                                {insight.title}
                                            </h4>
                                            <p className="text-sm text-text-secondary">
                                                {insight.description}
                                            </p>
                                        </div>
                                        {insight.actionable && (
                                            <div className="ml-3 text-signal-green text-xs font-mono">
                                                ✓ Actionable
                                            </div>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Bundle Opportunities */}
                    {result.analysis.bundleOpportunities.length > 0 && (
                        <div className="space-y-3">
                            <h3 className="font-mono text-sm uppercase tracking-wider text-text-tertiary">
                                Bundle Opportunities
                            </h3>
                            {result.analysis.bundleOpportunities.map((bundle, idx) => (
                                <div key={idx} className="hud-panel p-4 border-l-4 border-signal-amber">
                                    <div className="flex items-start justify-between mb-3">
                                        <div>
                                            <h4 className="font-mono font-bold text-text-primary mb-2">
                                                Bundle #{idx + 1}
                                            </h4>
                                            <div className="space-y-1 text-sm text-text-secondary">
                                                {bundle.products.map((prod, i) => (
                                                    <div key={i}>• {prod}</div>
                                                ))}
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="text-xs text-text-tertiary mb-1">Bundle Price</div>
                                            <div className="text-2xl font-bold text-signal-green">
                                                ₹{bundle.expectedValue}
                                            </div>
                                            <div className="text-xs text-signal-amber mt-1">
                                                Save {bundle.discount}%
                                            </div>
                                        </div>
                                    </div>
                                    <p className="text-sm text-text-secondary italic">
                                        {bundle.reason}
                                    </p>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Powered By */}
                    <div className="hud-panel p-3 flex items-center justify-center gap-4 text-xs font-mono text-text-tertiary">
                        <span>Powered by:</span>
                        <span className="text-signal-cyan">TensorFlow.js ({result.powered.backend})</span>
                        {result.powered.gemini && (
                            <span className="text-signal-amber">+ Gemini AI with Code Execution</span>
                        )}
                    </div>
                </div>
            )}

            {/* Empty State */}
            {!result && !isLoading && (
                <div className="flex-1 flex items-center justify-center">
                    <div className="text-center max-w-md">
                        <div className="text-6xl mb-4">🤖</div>
                        <h3 className="font-mono text-lg font-bold text-text-primary mb-2">
                            Smart Recommendations Engine
                        </h3>
                        <p className="text-sm text-text-secondary mb-6">
                            Select a product and click "Generate" to see TensorFlow-powered product recommendations combined with Gemini AI analysis.
                        </p>
                        <div className="space-y-2 text-xs text-text-tertiary text-left bg-surface-elevated p-4 border border-border-default">
                            <div>✓ TensorFlow.js product embeddings</div>
                            <div>✓ Cosine similarity matching</div>
                            <div>✓ Gemini AI strategic analysis</div>
                            <div>✓ Code execution for pricing optimization</div>
                            <div>✓ Zero training data required</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
