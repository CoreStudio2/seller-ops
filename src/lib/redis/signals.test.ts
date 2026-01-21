import { describe, it, expect } from 'vitest';
import {
    createSignal,
    serializeSignal,
    deserializeSignal,
    detectThreatsFromSignal,
    generateMockSignal,
    CHANNELS,
    KEYS,
} from '@/lib/redis/signals';
import type { Signal, SignalId } from '@/lib/types';

describe('Redis Signal Client', () => {
    describe('createSignal', () => {
        it('should create a signal with required fields', () => {
            const signal = createSignal('PRICE_CHANGE', 100);

            expect(signal.id).toBeDefined();
            expect(signal.type).toBe('PRICE_CHANGE');
            expect(signal.value).toBe(100);
            expect(signal.timestamp).toBeInstanceOf(Date);
        });

        it('should calculate delta from previousValue', () => {
            const signal = createSignal('PRICE_CHANGE', 90, { previousValue: 100 });

            expect(signal.delta).toBe(-10);
        });

        it('should use provided delta over calculated', () => {
            const signal = createSignal('PRICE_CHANGE', 90, {
                previousValue: 100,
                delta: -15, // Override
            });

            expect(signal.delta).toBe(-15);
        });

        it('should include optional metadata', () => {
            const signal = createSignal('COMPETITOR_PRICE', 85, {
                metadata: { competitor: 'Brand X', source: 'scraper' },
            });

            expect(signal.metadata).toEqual({ competitor: 'Brand X', source: 'scraper' });
        });
    });

    describe('serializeSignal / deserializeSignal', () => {
        it('should roundtrip a signal correctly', () => {
            const original = createSignal('REFUND_SPIKE', 12.5, { previousValue: 3 });

            const serialized = serializeSignal(original);
            const deserialized = deserializeSignal(serialized);

            expect(deserialized.id).toBe(original.id);
            expect(deserialized.type).toBe(original.type);
            expect(deserialized.value).toBe(original.value);
            expect(deserialized.timestamp.getTime()).toBe(original.timestamp.getTime());
        });

        it('should produce valid JSON', () => {
            const signal = createSignal('STOCK_ALERT', 5);
            const serialized = serializeSignal(signal);

            expect(() => JSON.parse(serialized)).not.toThrow();
        });
    });

    describe('detectThreatsFromSignal', () => {
        it('should detect CRITICAL refund spike', () => {
            const signal = createSignal('REFUND_SPIKE', 8); // > 5%

            const threat = detectThreatsFromSignal(signal);

            expect(threat).not.toBeNull();
            expect(threat?.severity).toBe('CRITICAL');
            expect(threat?.shouldAlert).toBe(true);
        });

        it('should not alert for normal refund rate', () => {
            const signal = createSignal('REFUND_SPIKE', 2); // < 5%

            const threat = detectThreatsFromSignal(signal);

            expect(threat).toBeNull();
        });

        it('should detect conversion drop', () => {
            const signal = createSignal('CONVERSION_DROP', 2, { previousValue: 4 }); // 50% drop

            const threat = detectThreatsFromSignal(signal);

            expect(threat).not.toBeNull();
            expect(threat?.severity).toBe('WARNING');
        });

        it('should detect competitor price undercut', () => {
            const signal = createSignal('COMPETITOR_PRICE', 85, { previousValue: 100 });

            const threat = detectThreatsFromSignal(signal);

            expect(threat).not.toBeNull();
            expect(threat?.severity).toBe('CRITICAL');
        });

        it('should detect low stock', () => {
            const signal = createSignal('STOCK_ALERT', 5); // < 10

            const threat = detectThreatsFromSignal(signal);

            expect(threat).not.toBeNull();
            expect(threat?.severity).toBe('WARNING');
        });
    });

    describe('generateMockSignal', () => {
        it('should generate valid signals', () => {
            const signal = generateMockSignal();

            expect(signal.id).toBeDefined();
            expect(signal.type).toBeDefined();
            expect(typeof signal.value).toBe('number');
            expect(signal.timestamp).toBeInstanceOf(Date);
        });

        it('should generate different signals', () => {
            const signals = Array.from({ length: 10 }, () => generateMockSignal());
            const uniqueTypes = new Set(signals.map(s => s.type));

            // Should have some variety
            expect(uniqueTypes.size).toBeGreaterThan(1);
        });
    });

    describe('CHANNELS and KEYS', () => {
        it('should have all required channel names', () => {
            expect(CHANNELS.THREATS).toBe('sellerops:threats');
            expect(CHANNELS.SIGNALS).toBe('sellerops:signals');
            expect(CHANNELS.STATUS).toBe('sellerops:status');
        });

        it('should have all required key patterns', () => {
            expect(KEYS.SIGNAL_PREFIX).toBe('signal:');
            expect(KEYS.THREAT_PREFIX).toBe('threat:');
            expect(KEYS.STATUS).toBe('live:status');
        });
    });
});
