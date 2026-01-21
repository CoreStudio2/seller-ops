import Redis from 'ioredis';
import { CHANNELS, KEYS } from './signals';

// === CONFIGURATION ===
const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';

// === SINGLETON CLIENT ===
let redisClient: Redis | null = null;

export function getRedisClient(): Redis {
    if (!redisClient) {
        redisClient = new Redis(REDIS_URL, {
            maxRetriesPerRequest: 3,
            retryStrategy(times) {
                const delay = Math.min(times * 50, 2000);
                return delay;
            },
            // Don't crash on connection error, just log (resilience)
            reconnectOnError: (err) => {
                const targetError = 'READONLY';
                if (err.message.includes(targetError)) {
                    return true; // Reconnect on specific errors
                }
                return false;
            },
        });

        redisClient.on('error', (err) => {
            // Prevent unhandled exception crashing the server
            console.warn('[Redis] Connection error, retrying...', err.message);
        });

        redisClient.on('connect', () => {
            console.log('[Redis] Connected to War Room signal stream');
        });
    }
    return redisClient;
}

// === SIGNAL STREAMING ===

/**
 * Publishes a signal to the Redis stream and pub/sub channel
 */
export async function publishSignal(signalJson: string): Promise<void> {
    const client = getRedisClient();
    const pipeline = client.pipeline();

    // 1. Publish to Pub/Sub for immediate UI updates (via WebSocket/SSE if we had them)
    pipeline.publish(CHANNELS.SIGNALS, signalJson);

    // 2. Add to Stream/List for processing history (capped at 1000 items)
    pipeline.lpush(KEYS.SIGNAL_QUEUE, signalJson);
    pipeline.ltrim(KEYS.SIGNAL_QUEUE, 0, 999);

    await pipeline.exec();
}

/**
 * Gets the latest signals from the queue
 */
export async function getRecentSignals(limit: number = 50): Promise<string[]> {
    const client = getRedisClient();
    return client.lrange(KEYS.SIGNAL_QUEUE, 0, limit - 1);
}

/**
 * Updates the live status cache
 */
export async function updateLiveStatus(statusJson: string): Promise<void> {
    const client = getRedisClient();
    await client.set(KEYS.STATUS, statusJson, 'EX', 60); // Expire after 60s (force refresh)
}

/**
 * Gets the current live status
 */
export async function getCachedLiveStatus(): Promise<string | null> {
    const client = getRedisClient();
    return client.get(KEYS.STATUS);
}
