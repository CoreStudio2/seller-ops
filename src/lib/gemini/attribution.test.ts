/**
 * Gemini Attribution Core Integration Tests
 * These tests require a valid GEMINI_API_KEY in .env
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { generateAttributionBrief } from '@/lib/gemini/attribution';
import type { ThreatEvent, Signal, ThreatId, SignalId } from '@/lib/types';

// Skip tests if no API key (for CI environments)
const hasApiKey = !!process.env.GEMINI_API_KEY;

describe.skipIf(!hasApiKey)('Gemini Attribution Core - Integration', () => {

    // Test data
    const mockThreat: ThreatEvent = {
        id: 'test-threat-1' as ThreatId,
        severity: 'CRITICAL',
        title: 'Buy Box Lost - SKU #334',
        description: 'Competitor undercut price by ₹120. Current price ₹1,499 vs competitor ₹1,379.',
        signals: ['sig-1', 'sig-2'] as SignalId[],
        detectedAt: new Date(),
        suggestedAction: 'Match price to regain Buy Box',
        confidence: 94,
    };

    const mockSignals: Signal[] = [
        {
            id: 'sig-1' as SignalId,
            type: 'COMPETITOR_PRICE',
            timestamp: new Date(Date.now() - 3600000),
            value: 1379,
            previousValue: 1499,
            delta: -120,
        },
        {
            id: 'sig-2' as SignalId,
            type: 'CONVERSION_DROP',
            timestamp: new Date(Date.now() - 1800000),
            value: 2.1,
            previousValue: 3.8,
            delta: -1.7,
        },
        {
            id: 'sig-3' as SignalId,
            type: 'SEARCH_DROP',
            timestamp: new Date(Date.now() - 900000),
            value: 45,
            previousValue: 78,
            delta: -33,
        },
    ];

    it('should generate an attribution brief from Gemini', async () => {
        const brief = await generateAttributionBrief(mockThreat, mockSignals);

        // Verify structure
        expect(brief).toBeDefined();
        expect(brief.threatId).toBe(mockThreat.id);
        expect(brief.summary).toBeTruthy();
        expect(brief.summary.length).toBeGreaterThan(50);
        expect(brief.causes).toBeInstanceOf(Array);
        expect(brief.suggestedActions).toBeInstanceOf(Array);
        expect(brief.confidence).toBeGreaterThanOrEqual(0);
        expect(brief.confidence).toBeLessThanOrEqual(100);
        expect(brief.generatedAt).toBeInstanceOf(Date);
    }, 30000); // 30s timeout for API call

    it('should include causal factors', async () => {
        const brief = await generateAttributionBrief(mockThreat, mockSignals);

        expect(brief.causes.length).toBeGreaterThan(0);

        for (const cause of brief.causes) {
            expect(cause.factor).toBeTruthy();
            expect(['HIGH', 'MEDIUM', 'LOW']).toContain(cause.impact);
            expect(cause.evidence).toBeTruthy();
        }
    }, 30000);

    it('should include suggested actions', async () => {
        const brief = await generateAttributionBrief(mockThreat, mockSignals);

        expect(brief.suggestedActions.length).toBeGreaterThan(0);

        for (const action of brief.suggestedActions) {
            expect(action.action).toBeTruthy();
            expect(['IMMEDIATE', 'SOON', 'OPTIONAL']).toContain(action.priority);
            expect(action.expectedOutcome).toBeTruthy();
        }
    }, 30000);

    it('should reference signals in evidence', async () => {
        const brief = await generateAttributionBrief(mockThreat, mockSignals);

        // At least one cause should reference the data from signals
        const allEvidence = brief.causes.map(c => c.evidence.toLowerCase()).join(' ');
        const allSummary = brief.summary.toLowerCase();

        // Should mention price, conversion, or competitor somewhere
        const relevantTerms = ['price', 'competitor', 'conversion', 'drop', 'decrease'];
        const hasRelevantTerm = relevantTerms.some(
            term => allEvidence.includes(term) || allSummary.includes(term)
        );

        expect(hasRelevantTerm).toBe(true);
    }, 30000);

    it('should handle minimal signals gracefully', async () => {
        const minimalSignals: Signal[] = [
            {
                id: 'sig-minimal' as SignalId,
                type: 'PRICE_CHANGE',
                timestamp: new Date(),
                value: 100,
            },
        ];

        const brief = await generateAttributionBrief(mockThreat, minimalSignals);

        expect(brief).toBeDefined();
        expect(brief.summary).toBeTruthy();
    }, 30000);
});

// Basic unit tests that don't require API
describe('Gemini Attribution Core - Unit', () => {
    it('should have the module exports', async () => {
        const module = await import('@/lib/gemini/attribution');

        expect(module.generateAttributionBrief).toBeDefined();
        expect(typeof module.generateAttributionBrief).toBe('function');
        expect(module.streamAttributionBrief).toBeDefined();
        expect(typeof module.streamAttributionBrief).toBe('function');
    });
});
