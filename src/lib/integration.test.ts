import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ingestSignal, getDashboardData } from '@/lib/data';
import { createSignal } from '@/lib/redis/signals';

// Mock everything "Real"
vi.mock('@/lib/redis/client', () => ({
    getRedisClient: vi.fn(() => ({
        publish: vi.fn(),
        on: vi.fn(),
    })),
    publishSignal: vi.fn(),
    getRecentSignals: vi.fn(async () => []),
    updateLiveStatus: vi.fn(),
    getCachedLiveStatus: vi.fn(async () => null),
}));

vi.mock('@/lib/turso/database', () => ({
    saveSignal: vi.fn(async () => { }),
    saveThreat: vi.fn(async () => { }),
    getRecentThreats: vi.fn(async () => []),
}));

describe('Integration: Data Service', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('should ingest valid signal and publish to redis', async () => {
        const result = await ingestSignal('PRICE_CHANGE', 1500);

        expect(result.signal).toBeDefined();
        // Should publish to Redis (Hot path)
        const { publishSignal } = await import('@/lib/redis/client');
        expect(publishSignal).toHaveBeenCalled();
    });

    it('should detect threats and save them', async () => {
        // Competitor Price Undercut should trigger threat
        const result = await ingestSignal('COMPETITOR_PRICE', 500, { previousValue: 1000 });

        expect(result.threat).toBeDefined();
        expect(result.threat?.severity).toBe('CRITICAL');

        // Should save threat
        const { saveThreat } = await import('@/lib/turso/database');
        expect(saveThreat).toHaveBeenCalled();
    });

    it('should generate baseline dashboard data if cache miss', async () => {
        const data = await getDashboardData();

        expect(data.status).toBeDefined();
        expect(data.status.revenueVelocity).toBeGreaterThan(0);

        // Should update cache
        const { updateLiveStatus } = await import('@/lib/redis/client');
        expect(updateLiveStatus).toHaveBeenCalled();
    });
});
