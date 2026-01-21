import { describe, it, expect, expectTypeOf } from 'vitest';
import type {
    SignalId,
    ThreatId,
    SKUId,
    SignalType,
    SeverityLevel,
    Signal,
    ThreatEvent,
    AttributionBrief,
    SimulationParams,
    SimulationResult,
    LiveStatus,
} from '@/lib/types';

describe('Type Definitions', () => {
    describe('Branded Types', () => {
        it('SignalId should be a branded string', () => {
            const id = 'sig-123' as SignalId;
            expectTypeOf(id).toMatchTypeOf<string>();
        });

        it('ThreatId should be a branded string', () => {
            const id = 'threat-456' as ThreatId;
            expectTypeOf(id).toMatchTypeOf<string>();
        });

        it('SKUId should be a branded string', () => {
            const id = 'sku-789' as SKUId;
            expectTypeOf(id).toMatchTypeOf<string>();
        });
    });

    describe('Signal', () => {
        it('should have required properties', () => {
            const signal: Signal = {
                id: 'sig-1' as SignalId,
                type: 'PRICE_CHANGE',
                timestamp: new Date(),
                value: 100,
            };

            expect(signal.id).toBeDefined();
            expect(signal.type).toBeDefined();
            expect(signal.timestamp).toBeInstanceOf(Date);
            expect(typeof signal.value).toBe('number');
        });

        it('should allow optional properties', () => {
            const signal: Signal = {
                id: 'sig-2' as SignalId,
                type: 'COMPETITOR_PRICE',
                timestamp: new Date(),
                value: 95,
                previousValue: 100,
                delta: -5,
                skuId: 'sku-123' as SKUId,
                metadata: { competitor: 'Brand X' },
            };

            expect(signal.previousValue).toBe(100);
            expect(signal.delta).toBe(-5);
            expect(signal.skuId).toBeDefined();
            expect(signal.metadata).toBeDefined();
        });
    });

    describe('ThreatEvent', () => {
        it('should have all required fields', () => {
            const threat: ThreatEvent = {
                id: 'threat-1' as ThreatId,
                severity: 'CRITICAL',
                title: 'Buy Box Lost',
                description: 'Competitor undercut price',
                signals: ['sig-1' as SignalId],
                detectedAt: new Date(),
                confidence: 95,
            };

            expect(threat.id).toBeDefined();
            expect(['CRITICAL', 'WARNING', 'INFO', 'OK']).toContain(threat.severity);
            expect(threat.title).toBeTruthy();
            expect(threat.description).toBeTruthy();
            expect(Array.isArray(threat.signals)).toBe(true);
            expect(threat.detectedAt).toBeInstanceOf(Date);
            expect(threat.confidence).toBeGreaterThanOrEqual(0);
            expect(threat.confidence).toBeLessThanOrEqual(100);
        });

        it('should allow optional suggestedAction', () => {
            const threat: ThreatEvent = {
                id: 'threat-2' as ThreatId,
                severity: 'WARNING',
                title: 'Test',
                description: 'Test',
                signals: [],
                detectedAt: new Date(),
                confidence: 70,
                suggestedAction: 'Lower price by 5%',
            };

            expect(threat.suggestedAction).toBe('Lower price by 5%');
        });
    });

    describe('SimulationParams', () => {
        it('should have all three adjustment parameters', () => {
            const params: SimulationParams = {
                priceChange: -5,
                adSpendChange: 20,
                shippingSpeedChange: -1,
            };

            expect(typeof params.priceChange).toBe('number');
            expect(typeof params.adSpendChange).toBe('number');
            expect(typeof params.shippingSpeedChange).toBe('number');
        });
    });

    describe('SimulationResult', () => {
        it('should contain all projected metrics', () => {
            const result: SimulationResult = {
                params: { priceChange: 0, adSpendChange: 0, shippingSpeedChange: 0 },
                projectedRevenue: 50000,
                projectedMargin: 22,
                projectedVolume: 500,
                competitorResponseProbability: 20,
                riskLevel: 'OK',
                timestamp: new Date(),
            };

            expect(result.projectedRevenue).toBeGreaterThan(0);
            expect(result.projectedMargin).toBeDefined();
            expect(result.projectedVolume).toBeDefined();
            expect(result.competitorResponseProbability).toBeGreaterThanOrEqual(0);
            expect(result.competitorResponseProbability).toBeLessThanOrEqual(100);
            expect(['CRITICAL', 'WARNING', 'INFO', 'OK']).toContain(result.riskLevel);
        });
    });

    describe('LiveStatus', () => {
        it('should contain real-time metrics', () => {
            const status: LiveStatus = {
                revenueVelocity: 3500,
                revenueDirection: 'UP',
                riskScore: 45,
                opportunityScore: 60,
                activeThreats: 2,
                lastUpdated: new Date(),
            };

            expect(status.revenueVelocity).toBeGreaterThanOrEqual(0);
            expect(['UP', 'DOWN', 'STABLE']).toContain(status.revenueDirection);
            expect(status.riskScore).toBeGreaterThanOrEqual(0);
            expect(status.opportunityScore).toBeGreaterThanOrEqual(0);
            expect(status.activeThreats).toBeGreaterThanOrEqual(0);
            expect(status.lastUpdated).toBeInstanceOf(Date);
        });
    });

    describe('AttributionBrief', () => {
        it('should have Gemini-generated fields', () => {
            const brief: AttributionBrief = {
                threatId: 'threat-1' as ThreatId,
                summary: 'Analysis of the threat...',
                causes: [
                    { factor: 'Price Drop', impact: 'HIGH', evidence: 'Signal data shows...' },
                ],
                suggestedActions: [
                    { action: 'Match price', priority: 'IMMEDIATE', expectedOutcome: 'Regain Buy Box' },
                ],
                confidence: 87,
                generatedAt: new Date(),
            };

            expect(brief.threatId).toBeDefined();
            expect(brief.summary).toBeTruthy();
            expect(brief.causes.length).toBeGreaterThan(0);
            expect(brief.suggestedActions.length).toBeGreaterThan(0);
            expect(brief.confidence).toBeGreaterThanOrEqual(0);
            expect(brief.confidence).toBeLessThanOrEqual(100);
            expect(brief.generatedAt).toBeInstanceOf(Date);
        });

        it('causes should have proper impact levels', () => {
            const causes = [
                { factor: 'Test', impact: 'HIGH' as const, evidence: 'Data' },
                { factor: 'Test2', impact: 'MEDIUM' as const, evidence: 'Data' },
                { factor: 'Test3', impact: 'LOW' as const, evidence: 'Data' },
            ];

            causes.forEach(cause => {
                expect(['HIGH', 'MEDIUM', 'LOW']).toContain(cause.impact);
            });
        });

        it('actions should have proper priority levels', () => {
            const actions = [
                { action: 'Act', priority: 'IMMEDIATE' as const, expectedOutcome: 'Result' },
                { action: 'Act2', priority: 'SOON' as const, expectedOutcome: 'Result' },
                { action: 'Act3', priority: 'OPTIONAL' as const, expectedOutcome: 'Result' },
            ];

            actions.forEach(action => {
                expect(['IMMEDIATE', 'SOON', 'OPTIONAL']).toContain(action.priority);
            });
        });
    });
});
