/**
 * Unified Redis Client
 * Supports both local ioredis (dev) and Upstash (production/Vercel)
 */

import { CHANNELS, KEYS } from './signals';

// Conditional imports based on environment
const isUpstashConfigured = !!(
  process.env.UPSTASH_REDIS_REST_URL && 
  process.env.UPSTASH_REDIS_REST_TOKEN
);

// Flag to determine which client to use
const USE_UPSTASH = isUpstashConfigured || process.env.NODE_ENV === 'production';

// === REDIS CLIENT WRAPPER ===

interface RedisClient {
  publishSignal(signalJson: string): Promise<void>;
  getRecentSignals(count: number): Promise<string[]>;
  updateLiveStatus(statusJson: string): Promise<void>;
  getCachedLiveStatus(): Promise<string | null>;
}

class LocalRedisClient implements RedisClient {
  private Redis: any;
  private client: any = null;

  constructor() {
    // Lazy load ioredis only if needed
    this.Redis = require('ioredis').default;
  }

  private getClient() {
    if (!this.client) {
      const REDIS_URL = process.env.REDIS_URL || 'redis://localhost:6379';
      this.client = new this.Redis(REDIS_URL, {
        maxRetriesPerRequest: 3,
        retryStrategy(times: number) {
          const delay = Math.min(times * 50, 2000);
          return delay;
        },
        reconnectOnError: (err: Error) => {
          if (err.message.includes('READONLY')) {
            return true;
          }
          return false;
        },
      });

      this.client.on('error', (err: Error) => {
        console.warn('[Redis] Connection error:', err.message);
      });

      this.client.on('connect', () => {
        console.log('[Redis] Connected to War Room signal stream');
      });
    }
    return this.client;
  }

  async publishSignal(signalJson: string): Promise<void> {
    try {
      const client = this.getClient();
      const pipeline = client.pipeline();
      pipeline.publish(CHANNELS.SIGNALS, signalJson);
      pipeline.lpush(KEYS.SIGNAL_QUEUE, signalJson);
      pipeline.ltrim(KEYS.SIGNAL_QUEUE, 0, 999);
      await pipeline.exec();
    } catch (error) {
      console.error('[LocalRedis] Publish failed:', error);
    }
  }

  async getRecentSignals(count: number): Promise<string[]> {
    try {
      const client = this.getClient();
      return await client.lrange(KEYS.SIGNAL_QUEUE, 0, count - 1);
    } catch (error) {
      console.error('[LocalRedis] Get signals failed:', error);
      return [];
    }
  }

  async updateLiveStatus(statusJson: string): Promise<void> {
    try {
      const client = this.getClient();
      await client.set(KEYS.STATUS, statusJson, 'EX', 60);
    } catch (error) {
      console.error('[LocalRedis] Update status failed:', error);
    }
  }

  async getCachedLiveStatus(): Promise<string | null> {
    try {
      const client = this.getClient();
      return await client.get(KEYS.STATUS);
    } catch (error) {
      console.error('[LocalRedis] Get status failed:', error);
      return null;
    }
  }
}

class UpstashRedisClient implements RedisClient {
  private Redis: any;
  private client: any = null;

  constructor() {
    // Lazy load @upstash/redis only if needed
    this.Redis = require('@upstash/redis').Redis;
  }

  private getClient() {
    if (!this.client) {
      this.client = new this.Redis({
        url: process.env.UPSTASH_REDIS_REST_URL!,
        token: process.env.UPSTASH_REDIS_REST_TOKEN!,
      });
    }
    return this.client;
  }

  async publishSignal(signalJson: string): Promise<void> {
    try {
      const client = this.getClient();
      // Upstash doesn't support pipelines, run sequentially
      await client.publish(CHANNELS.SIGNALS, signalJson);
      await client.lpush(KEYS.SIGNAL_QUEUE, signalJson);
      await client.ltrim(KEYS.SIGNAL_QUEUE, 0, 999);
    } catch (error) {
      console.error('[Upstash] Publish failed:', error);
    }
  }

  async getRecentSignals(count: number): Promise<string[]> {
    try {
      const client = this.getClient();
      const result = await client.lrange(KEYS.SIGNAL_QUEUE, 0, count - 1);
      return result || [];
    } catch (error) {
      console.error('[Upstash] Get signals failed:', error);
      return [];
    }
  }

  async updateLiveStatus(statusJson: string): Promise<void> {
    try {
      const client = this.getClient();
      await client.set(KEYS.STATUS, statusJson, { ex: 60 });
    } catch (error) {
      console.error('[Upstash] Update status failed:', error);
    }
  }

  async getCachedLiveStatus(): Promise<string | null> {
    try {
      const client = this.getClient();
      return await client.get(KEYS.STATUS);
    } catch (error) {
      console.error('[Upstash] Get status failed:', error);
      return null;
    }
  }
}

// Fallback client (no Redis, graceful degradation)
class FallbackRedisClient implements RedisClient {
  private inMemorySignals: string[] = [];
  private inMemoryStatus: string | null = null;

  async publishSignal(signalJson: string): Promise<void> {
    console.warn('[Fallback] Using in-memory signal storage (Redis unavailable)');
    this.inMemorySignals.unshift(signalJson);
    if (this.inMemorySignals.length > 1000) {
      this.inMemorySignals = this.inMemorySignals.slice(0, 1000);
    }
  }

  async getRecentSignals(count: number): Promise<string[]> {
    return this.inMemorySignals.slice(0, count);
  }

  async updateLiveStatus(statusJson: string): Promise<void> {
    this.inMemoryStatus = statusJson;
  }

  async getCachedLiveStatus(): Promise<string | null> {
    return this.inMemoryStatus;
  }
}

// === SINGLETON INSTANCE ===
let redisClient: RedisClient | null = null;

function initializeRedisClient(): RedisClient {
  if (USE_UPSTASH) {
    console.log('[Redis] Using Upstash (serverless mode)');
    return new UpstashRedisClient();
  } else if (process.env.REDIS_URL || process.env.NODE_ENV === 'development') {
    console.log('[Redis] Using local ioredis');
    return new LocalRedisClient();
  } else {
    console.warn('[Redis] No Redis configured, using fallback mode');
    return new FallbackRedisClient();
  }
}

export function getRedisClient(): RedisClient {
  if (!redisClient) {
    redisClient = initializeRedisClient();
  }
  return redisClient;
}

// === PUBLIC API ===

export async function publishSignal(signalJson: string): Promise<void> {
  const client = getRedisClient();
  await client.publishSignal(signalJson);
}

export async function getRecentSignals(count: number = 10): Promise<string[]> {
  const client = getRedisClient();
  return await client.getRecentSignals(count);
}

export async function updateLiveStatus(statusJson: string): Promise<void> {
  const client = getRedisClient();
  await client.updateLiveStatus(statusJson);
}

export async function getCachedLiveStatus(): Promise<string | null> {
  const client = getRedisClient();
  return await client.getCachedLiveStatus();
}

// === UTILITY ===

export function getRedisMode(): 'upstash' | 'local' | 'fallback' {
  if (USE_UPSTASH) return 'upstash';
  if (process.env.REDIS_URL || process.env.NODE_ENV === 'development') return 'local';
  return 'fallback';
}
