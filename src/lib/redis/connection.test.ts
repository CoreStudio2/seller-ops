import { describe, it, expect } from 'vitest';
import Redis from 'ioredis';

// REAL Redis test (requires Docker container running)
describe('Redis Connection (Real)', () => {
    it('should connect to local Redis', async () => {
        const redis = new Redis('redis://localhost:6379', {
            maxRetriesPerRequest: 1,
            retryStrategy: () => null // Don't retry, fail fast
        });

        try {
            await redis.set('test-key', 'hello-war-room');
            const value = await redis.get('test-key');
            expect(value).toBe('hello-war-room');
        } finally {
            await redis.quit();
        }
    });
});
