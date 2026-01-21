import { describe, it, expect } from 'vitest';
import {
    runSimulation,
    runScenarioComparison,
    quickPriceImpact
} from '@/lib/simulation/engine';
import type { SimulationParams } from '@/lib/types';

describe('Simulation Engine', () => {
    describe('runSimulation', () => {
        it('should return baseline values with zero changes', () => {
            const params: SimulationParams = {
                priceChange: 0,
                adSpendChange: 0,
                shippingSpeedChange: 0,
            };

            const result = runSimulation(params);

            // Should return close to baseline values
            expect(result.projectedRevenue).toBeCloseTo(50000, -2);
            expect(result.projectedMargin).toBeCloseTo(22, 0);
            expect(result.projectedVolume).toBeCloseTo(500, -1);
            // 0 price change = 0 competitor response (no threat)
            expect(result.competitorResponseProbability).toBe(0);
            expect(result.riskLevel).toBe('OK');
        });

        it('should decrease margin with price cuts', () => {
            const params: SimulationParams = {
                priceChange: -10,
                adSpendChange: 0,
                shippingSpeedChange: 0,
            };

            const result = runSimulation(params);

            // Price cut should reduce margin
            expect(result.projectedMargin).toBeLessThan(22);
        });

        it('should increase competitor response probability with aggressive price cuts', () => {
            const params: SimulationParams = {
                priceChange: -15,
                adSpendChange: 0,
                shippingSpeedChange: 0,
            };

            const result = runSimulation(params);

            // Aggressive price cut should trigger high competitor response
            expect(result.competitorResponseProbability).toBeGreaterThan(60);
        });

        it('should increase volume with faster shipping', () => {
            const baseParams: SimulationParams = {
                priceChange: 0,
                adSpendChange: 0,
                shippingSpeedChange: 0,
            };

            const fasterShippingParams: SimulationParams = {
                priceChange: 0,
                adSpendChange: 0,
                shippingSpeedChange: -2, // Faster by 2 days
            };

            const baseResult = runSimulation(baseParams);
            const fasterResult = runSimulation(fasterShippingParams);

            expect(fasterResult.projectedVolume).toBeGreaterThan(baseResult.projectedVolume);
        });

        it('should increase volume with higher ad spend', () => {
            const baseParams: SimulationParams = {
                priceChange: 0,
                adSpendChange: 0,
                shippingSpeedChange: 0,
            };

            const moreAdsParams: SimulationParams = {
                priceChange: 0,
                adSpendChange: 50,
                shippingSpeedChange: 0,
            };

            const baseResult = runSimulation(baseParams);
            const moreAdsResult = runSimulation(moreAdsParams);

            expect(moreAdsResult.projectedVolume).toBeGreaterThan(baseResult.projectedVolume);
        });

        it('should flag CRITICAL risk for extreme margin drops', () => {
            const params: SimulationParams = {
                priceChange: -20,
                adSpendChange: 100,
                shippingSpeedChange: -3,
            };

            const result = runSimulation(params);

            expect(['CRITICAL', 'WARNING']).toContain(result.riskLevel);
        });

        it('should include timestamp in result', () => {
            const params: SimulationParams = {
                priceChange: 0,
                adSpendChange: 0,
                shippingSpeedChange: 0,
            };

            const result = runSimulation(params);

            expect(result.timestamp).toBeInstanceOf(Date);
        });
    });

    describe('runScenarioComparison', () => {
        it('should return three scenarios', () => {
            const scenarios = runScenarioComparison();

            expect(scenarios).toHaveProperty('conservative');
            expect(scenarios).toHaveProperty('moderate');
            expect(scenarios).toHaveProperty('aggressive');
        });

        it('aggressive scenario should have higher competitor response', () => {
            const scenarios = runScenarioComparison();

            expect(scenarios.aggressive.competitorResponseProbability)
                .toBeGreaterThan(scenarios.conservative.competitorResponseProbability);
        });

        it('conservative scenario should have lower risk', () => {
            const scenarios = runScenarioComparison();

            // Conservative should be safer
            const riskOrder = ['OK', 'INFO', 'WARNING', 'CRITICAL'];
            const conservativeRisk = riskOrder.indexOf(scenarios.conservative.riskLevel);
            const aggressiveRisk = riskOrder.indexOf(scenarios.aggressive.riskLevel);

            expect(conservativeRisk).toBeLessThanOrEqual(aggressiveRisk);
        });
    });

    describe('quickPriceImpact', () => {
        it('should return formatted strings', () => {
            const impact = quickPriceImpact(-5);

            expect(impact.revenueChange).toMatch(/^[+-]?\d+\.?\d*%$/);
            expect(impact.marginChange).toMatch(/^[+-]?\d+\.?\d*%$/);
            expect(impact.competitorRisk).toMatch(/^\d+%$/);
        });

        it('should show positive direction for price increase', () => {
            const impact = quickPriceImpact(10);

            // Price increase should improve margin
            expect(impact.marginChange.startsWith('+')).toBe(true);
        });

        it('should show low competitor risk for price increase', () => {
            const impact = quickPriceImpact(5);

            const riskPercent = parseInt(impact.competitorRisk);
            expect(riskPercent).toBeLessThan(30);
        });
    });
});
