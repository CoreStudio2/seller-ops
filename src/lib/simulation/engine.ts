/**
 * Beast Mode Simulation Engine
 * Numeric game-theory simulations for price/margin trade-offs
 * Uses deterministic formulas (no ML needed for demo)
 */

import type { SimulationParams, SimulationResult, SeverityLevel } from '@/lib/types';

// === BASELINE METRICS (Mock current state) ===
interface BaselineMetrics {
    currentRevenue: number;      // Daily revenue
    currentMargin: number;       // Margin percentage
    currentVolume: number;       // Units sold per day
    currentPrice: number;        // Average selling price
    competitorPrice: number;     // Competitor average price
    elasticity: number;          // Price elasticity of demand
}

// Default baseline for demo
const DEFAULT_BASELINE: BaselineMetrics = {
    currentRevenue: 50000,
    currentMargin: 22,
    currentVolume: 500,
    currentPrice: 100,
    competitorPrice: 95,
    elasticity: -1.5, // Elastic demand
};

// === SIMULATION FORMULAS ===

/**
 * Calculate projected volume based on price change
 * Uses price elasticity of demand: %ΔQ = elasticity × %ΔP
 */
function calculateVolumeChange(
    priceChangePercent: number,
    elasticity: number
): number {
    const volumeChangePercent = elasticity * priceChangePercent;
    return volumeChangePercent;
}

/**
 * Calculate competitor response probability
 * Based on game theory - competitors more likely to respond to aggressive cuts
 */
function calculateCompetitorResponse(priceChangePercent: number): number {
    if (priceChangePercent >= 0) {
        // Price increase - low response probability
        return Math.min(20, priceChangePercent * 2);
    }
    // Price cut - higher response probability
    const magnitude = Math.abs(priceChangePercent);
    if (magnitude <= 3) return 20 + magnitude * 5;
    if (magnitude <= 7) return 35 + magnitude * 4;
    return Math.min(95, 50 + magnitude * 3);
}

/**
 * Determine risk level based on projected outcomes
 */
function calculateRiskLevel(
    marginChange: number,
    competitorResponse: number
): SeverityLevel {
    if (marginChange < -10 || competitorResponse > 80) return 'CRITICAL';
    if (marginChange < -5 || competitorResponse > 60) return 'WARNING';
    if (marginChange < 0 || competitorResponse > 40) return 'INFO';
    return 'OK';
}

// === MAIN SIMULATION FUNCTION ===

export function runSimulation(
    params: SimulationParams,
    baseline: BaselineMetrics = DEFAULT_BASELINE
): SimulationResult {
    const { priceChange, adSpendChange, shippingSpeedChange } = params;

    // 1. Calculate new price
    const newPrice = baseline.currentPrice * (1 + priceChange / 100);

    // 2. Calculate volume impact from price change
    const volumeFromPrice = calculateVolumeChange(priceChange, baseline.elasticity);

    // 3. Calculate volume impact from ad spend (diminishing returns)
    const volumeFromAds = adSpendChange > 0
        ? Math.log(1 + adSpendChange / 10) * 5
        : adSpendChange / 2;

    // 4. Calculate volume impact from shipping speed (faster = more sales)
    const volumeFromShipping = shippingSpeedChange < 0
        ? Math.abs(shippingSpeedChange) * 3
        : shippingSpeedChange * -2;

    // 5. Total volume change
    const totalVolumeChange = volumeFromPrice + volumeFromAds + volumeFromShipping;
    const newVolume = baseline.currentVolume * (1 + totalVolumeChange / 100);

    // 6. Calculate new margin (affected by ad spend and shipping)
    const marginImpactFromAds = adSpendChange * 0.1; // Ads reduce margin
    const marginImpactFromShipping = shippingSpeedChange < 0
        ? Math.abs(shippingSpeedChange) * 0.5  // Faster shipping costs more
        : shippingSpeedChange * 0.3;

    const newMargin = baseline.currentMargin
        - marginImpactFromAds
        - marginImpactFromShipping
        + (priceChange > 0 ? priceChange * 0.3 : priceChange * 0.1);

    // 7. Calculate projected revenue
    const projectedRevenue = newPrice * newVolume;

    // 8. Competitor response probability
    const competitorResponse = calculateCompetitorResponse(priceChange);

    // 9. Risk assessment
    const marginChange = newMargin - baseline.currentMargin;
    const riskLevel = calculateRiskLevel(marginChange, competitorResponse);

    return {
        params,
        projectedRevenue: Math.round(projectedRevenue * 100) / 100,
        projectedMargin: Math.round(newMargin * 100) / 100,
        projectedVolume: Math.round(newVolume),
        competitorResponseProbability: Math.round(competitorResponse),
        riskLevel,
        timestamp: new Date(),
    };
}

// === BATCH SIMULATION (For scenario comparison) ===

export interface ScenarioComparison {
    conservative: SimulationResult;
    moderate: SimulationResult;
    aggressive: SimulationResult;
}

export function runScenarioComparison(
    baseline: BaselineMetrics = DEFAULT_BASELINE
): ScenarioComparison {
    return {
        conservative: runSimulation({ priceChange: -2, adSpendChange: 0, shippingSpeedChange: 0 }, baseline),
        moderate: runSimulation({ priceChange: -5, adSpendChange: 10, shippingSpeedChange: -1 }, baseline),
        aggressive: runSimulation({ priceChange: -10, adSpendChange: 25, shippingSpeedChange: -2 }, baseline),
    };
}

// === WHAT-IF QUICK CALC ===

export function quickPriceImpact(priceChangePercent: number): {
    revenueChange: string;
    marginChange: string;
    competitorRisk: string;
} {
    const result = runSimulation({
        priceChange: priceChangePercent,
        adSpendChange: 0,
        shippingSpeedChange: 0
    });

    const baseRevenue = DEFAULT_BASELINE.currentRevenue;
    const revenueChangePercent = ((result.projectedRevenue - baseRevenue) / baseRevenue) * 100;
    const marginChangeNum = result.projectedMargin - DEFAULT_BASELINE.currentMargin;

    return {
        revenueChange: `${revenueChangePercent >= 0 ? '+' : ''}${revenueChangePercent.toFixed(1)}%`,
        marginChange: `${marginChangeNum >= 0 ? '+' : ''}${marginChangeNum.toFixed(1)}%`,
        competitorRisk: `${result.competitorResponseProbability}%`,
    };
}
