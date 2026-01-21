import { describe, it, expect, beforeEach } from 'vitest';
import { useSellerOpsStore } from '@/lib/store';
import type { ThreatEvent, LiveStatus, AttributionBrief, ThreatId, SignalId } from '@/lib/types';

describe('SellerOps Store', () => {
    // Reset store before each test
    beforeEach(() => {
        useSellerOpsStore.setState({
            liveStatus: null,
            threats: [],
            selectedThreat: null,
            currentBrief: null,
            isLoadingBrief: false,
            simulationResult: null,
            isSimulating: false,
        });
    });

    describe('setLiveStatus', () => {
        it('should update live status', () => {
            const status: LiveStatus = {
                revenueVelocity: 5000,
                revenueDirection: 'UP',
                riskScore: 30,
                opportunityScore: 70,
                activeThreats: 1,
                lastUpdated: new Date(),
            };

            useSellerOpsStore.getState().setLiveStatus(status);

            expect(useSellerOpsStore.getState().liveStatus).toEqual(status);
        });
    });

    describe('addThreat', () => {
        it('should add a threat to the beginning of the list', () => {
            const threat: ThreatEvent = {
                id: 'threat-1' as ThreatId,
                severity: 'CRITICAL',
                title: 'Test Threat',
                description: 'Test description',
                signals: ['sig-1' as SignalId],
                detectedAt: new Date(),
                confidence: 90,
            };

            useSellerOpsStore.getState().addThreat(threat);

            const { threats } = useSellerOpsStore.getState();
            expect(threats).toHaveLength(1);
            expect(threats[0]).toEqual(threat);
        });

        it('should keep newest threats first', () => {
            const threat1: ThreatEvent = {
                id: 'threat-1' as ThreatId,
                severity: 'WARNING',
                title: 'First Threat',
                description: 'First',
                signals: [],
                detectedAt: new Date(),
                confidence: 80,
            };

            const threat2: ThreatEvent = {
                id: 'threat-2' as ThreatId,
                severity: 'CRITICAL',
                title: 'Second Threat',
                description: 'Second',
                signals: [],
                detectedAt: new Date(),
                confidence: 95,
            };

            const store = useSellerOpsStore.getState();
            store.addThreat(threat1);
            store.addThreat(threat2);

            const { threats } = useSellerOpsStore.getState();
            expect(threats[0].id).toBe('threat-2');
            expect(threats[1].id).toBe('threat-1');
        });

        it('should limit threats to 50', () => {
            const store = useSellerOpsStore.getState();

            // Add 55 threats
            for (let i = 0; i < 55; i++) {
                store.addThreat({
                    id: `threat-${i}` as ThreatId,
                    severity: 'INFO',
                    title: `Threat ${i}`,
                    description: `Description ${i}`,
                    signals: [],
                    detectedAt: new Date(),
                    confidence: 50,
                });
            }

            const { threats } = useSellerOpsStore.getState();
            expect(threats).toHaveLength(50);
        });
    });

    describe('selectThreat', () => {
        it('should set selected threat', () => {
            const threat: ThreatEvent = {
                id: 'threat-1' as ThreatId,
                severity: 'WARNING',
                title: 'Test',
                description: 'Test',
                signals: [],
                detectedAt: new Date(),
                confidence: 75,
            };

            useSellerOpsStore.getState().selectThreat(threat);

            expect(useSellerOpsStore.getState().selectedThreat).toEqual(threat);
        });

        it('should clear attribution brief when selecting new threat', () => {
            const brief: AttributionBrief = {
                threatId: 'threat-1' as ThreatId,
                summary: 'Test summary',
                causes: [],
                suggestedActions: [],
                confidence: 80,
                generatedAt: new Date(),
            };

            // Set a brief first
            useSellerOpsStore.setState({ currentBrief: brief });

            // Select a new threat
            useSellerOpsStore.getState().selectThreat({
                id: 'threat-2' as ThreatId,
                severity: 'CRITICAL',
                title: 'New Threat',
                description: 'New',
                signals: [],
                detectedAt: new Date(),
                confidence: 90,
            });

            expect(useSellerOpsStore.getState().currentBrief).toBeNull();
        });

        it('should allow deselecting by passing null', () => {
            const threat: ThreatEvent = {
                id: 'threat-1' as ThreatId,
                severity: 'INFO',
                title: 'Test',
                description: 'Test',
                signals: [],
                detectedAt: new Date(),
                confidence: 60,
            };

            const store = useSellerOpsStore.getState();
            store.selectThreat(threat);
            store.selectThreat(null);

            expect(useSellerOpsStore.getState().selectedThreat).toBeNull();
        });
    });

    describe('setAttributionBrief', () => {
        it('should set attribution brief and clear loading', () => {
            useSellerOpsStore.setState({ isLoadingBrief: true });

            const brief: AttributionBrief = {
                threatId: 'threat-1' as ThreatId,
                summary: 'Analysis complete',
                causes: [
                    { factor: 'Price', impact: 'HIGH', evidence: 'Data shows...' },
                ],
                suggestedActions: [
                    { action: 'Lower price', priority: 'IMMEDIATE', expectedOutcome: 'Regain market share' },
                ],
                confidence: 87,
                generatedAt: new Date(),
            };

            useSellerOpsStore.getState().setAttributionBrief(brief);

            const state = useSellerOpsStore.getState();
            expect(state.currentBrief).toEqual(brief);
            expect(state.isLoadingBrief).toBe(false);
        });
    });
});
