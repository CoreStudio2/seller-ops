/**
 * Upstash Redis Client for Vercel Serverless
 * REST-based Redis for serverless compatibility
 */

import { Redis as UpstashRedis } from '@upstash/redis';

let upstashClient: UpstashRedis | null = null;

/**
 * Get Upstash Redis client (singleton)
 */
export function getUpstashRedis(): UpstashRedis {
  if (!upstashClient) {
    const url = process.env.UPSTASH_REDIS_REST_URL;
    const token = process.env.UPSTASH_REDIS_REST_TOKEN;

    if (!url || !token) {
      throw new Error('Upstash Redis credentials not configured. Set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN');
    }

    upstashClient = new UpstashRedis({
      url,
      token,
    });
  }

  return upstashClient;
}

/**
 * Publish signal to Upstash Redis
 */
export async function publishSignalUpstash(channel: string, data: string): Promise<void> {
  try {
    const redis = getUpstashRedis();
    await redis.publish(channel, data);
  } catch (error) {
    console.error('[Upstash] Failed to publish signal:', error);
    // Graceful degradation - don't throw
  }
}

/**
 * Set key-value with optional TTL
 */
export async function setUpstash(key: string, value: string, ttlSeconds?: number): Promise<void> {
  try {
    const redis = getUpstashRedis();
    if (ttlSeconds) {
      await redis.set(key, value, { ex: ttlSeconds });
    } else {
      await redis.set(key, value);
    }
  } catch (error) {
    console.error(`[Upstash] Failed to set key ${key}:`, error);
  }
}

/**
 * Get value by key
 */
export async function getUpstash(key: string): Promise<string | null> {
  try {
    const redis = getUpstashRedis();
    return await redis.get<string>(key);
  } catch (error) {
    console.error(`[Upstash] Failed to get key ${key}:`, error);
    return null;
  }
}

/**
 * Push to list (queue)
 */
export async function lpushUpstash(key: string, value: string): Promise<void> {
  try {
    const redis = getUpstashRedis();
    await redis.lpush(key, value);
  } catch (error) {
    console.error(`[Upstash] Failed to lpush to ${key}:`, error);
  }
}

/**
 * Get list range
 */
export async function lrangeUpstash(key: string, start: number, stop: number): Promise<string[]> {
  try {
    const redis = getUpstashRedis();
    const result = await redis.lrange<string>(key, start, stop);
    return result || [];
  } catch (error) {
    console.error(`[Upstash] Failed to lrange ${key}:`, error);
    return [];
  }
}

/**
 * Trim list to specified size
 */
export async function ltrimUpstash(key: string, start: number, stop: number): Promise<void> {
  try {
    const redis = getUpstashRedis();
    await redis.ltrim(key, start, stop);
  } catch (error) {
    console.error(`[Upstash] Failed to ltrim ${key}:`, error);
  }
}

/**
 * Check if Upstash is configured
 */
export function isUpstashConfigured(): boolean {
  return !!(process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN);
}

/**
 * Test Upstash connection
 */
export async function testUpstashConnection(): Promise<boolean> {
  try {
    const redis = getUpstashRedis();
    await redis.set('test:connection', 'ok', { ex: 10 });
    const result = await redis.get('test:connection');
    return result === 'ok';
  } catch (error) {
    console.error('[Upstash] Connection test failed:', error);
    return false;
  }
}
