/**
 * Redis Signal Client
 * Real-time signal intake and caching for SellerOps
 * Uses Redis MCP server connection
 */

import type { Signal, SignalId, SignalType } from '@/lib/types';

// === SIGNAL CHANNEL NAMES ===
export const CHANNELS = {
    THREATS: 'sellerops:threats',
    SIGNALS: 'sellerops:signals',
    STATUS: 'sellerops:status',
} as const;

// === KEY PATTERNS ===
export const KEYS = {
    SIGNAL_PREFIX: 'signal:',
    THREAT_PREFIX: 'threat:',
    STATUS: 'live:status',
    SIGNAL_QUEUE: 'queue:signals',
} as const;

// === SIGNAL SERIALIZATION ===
export function serializeSignal(signal: Signal): string {
    return JSON.stringify({
        ...signal,
        timestamp: signal.timestamp.toISOString(),
    });
}

export function deserializeSignal(data: string): Signal {
    const parsed = JSON.parse(data);
    return {
        ...parsed,
        timestamp: new Date(parsed.timestamp),
    };
}

// === SIGNAL FACTORY ===
export function createSignal(
    type: SignalType,
    value: number,
    options?: {
        previousValue?: number;
        delta?: number;
        skuId?: string;
        metadata?: Record<string, unknown>;
    }
): Signal {
    return {
        id: `sig-${Date.now()}-${Math.random().toString(36).substring(7)}` as SignalId,
        type,
        timestamp: new Date(),
        value,
        previousValue: options?.previousValue,
        delta: options?.delta ?? (options?.previousValue !== undefined
            ? value - options.previousValue
            : undefined
        ),
        skuId: options?.skuId as Signal['skuId'],
        metadata: options?.metadata,
    };
}

// === SIGNAL DETECTION RULES ===
export interface SignalRule {
    type: SignalType;
    condition: (value: number, previousValue?: number) => boolean;
    severity: 'CRITICAL' | 'WARNING' | 'INFO';
    getMessage: (value: number, delta?: number) => string;
}

export const SIGNAL_RULES: SignalRule[] = [
    {
        type: 'PRICE_CHANGE',
        condition: (_, prev) => prev !== undefined,
        severity: 'WARNING',
        getMessage: (value, delta) =>
            `Price ${delta && delta > 0 ? 'increased' : 'decreased'} to ₹${value}`,
    },
    {
        type: 'REFUND_SPIKE',
        condition: (value) => value > 5, // More than 5% refund rate
        severity: 'CRITICAL',
        getMessage: (value) => `Refund rate spiked to ${value}%`,
    },
    {
        type: 'CONVERSION_DROP',
        condition: (value, prev) => prev !== undefined && value < prev * 0.8,
        severity: 'WARNING',
        getMessage: (value) => `Conversion dropped to ${value}%`,
    },
    {
        type: 'COMPETITOR_PRICE',
        condition: (value, prev) => prev !== undefined && value < prev,
        severity: 'CRITICAL',
        getMessage: (value) => `Competitor undercut to ₹${value}`,
    },
    {
        type: 'SEARCH_DROP',
        condition: (value, prev) => prev !== undefined && value < prev * 0.7,
        severity: 'WARNING',
        getMessage: (value) => `Search visibility dropped to ${value}%`,
    },
    {
        type: 'STOCK_ALERT',
        condition: (value) => value < 10, // Less than 10 units
        severity: 'WARNING',
        getMessage: (value) => `Low stock: ${value} units remaining`,
    },
];

// === DETECT THREATS FROM SIGNALS ===
export function detectThreatsFromSignal(signal: Signal): {
    shouldAlert: boolean;
    severity: 'CRITICAL' | 'WARNING' | 'INFO';
    message: string;
} | null {
    const rule = SIGNAL_RULES.find(r => r.type === signal.type);
    if (!rule) return null;

    if (rule.condition(signal.value, signal.previousValue)) {
        return {
            shouldAlert: true,
            severity: rule.severity,
            message: rule.getMessage(signal.value, signal.delta),
        };
    }

    return null;
}

// === MOCK SIGNAL GENERATOR (For demo) ===
export function generateMockSignal(): Signal {
    const types: SignalType[] = [
        'PRICE_CHANGE',
        'REFUND_SPIKE',
        'CONVERSION_DROP',
        'COMPETITOR_PRICE',
        'SEARCH_DROP',
    ];

    const type = types[Math.floor(Math.random() * types.length)];
    const baseValues: Record<SignalType, { min: number; max: number }> = {
        PRICE_CHANGE: { min: 800, max: 1500 },
        REFUND_SPIKE: { min: 1, max: 15 },
        CONVERSION_DROP: { min: 1, max: 5 },
        COMPETITOR_PRICE: { min: 700, max: 1400 },
        SEARCH_DROP: { min: 20, max: 100 },
        CART_ABANDONMENT: { min: 50, max: 90 },
        SHIPPING_DELAY: { min: 1, max: 7 },
        STOCK_ALERT: { min: 0, max: 50 },
    };

    const range = baseValues[type];
    const value = Math.floor(Math.random() * (range.max - range.min) + range.min);
    const previousValue = Math.floor(Math.random() * (range.max - range.min) + range.min);

    return createSignal(type, value, { previousValue });
}
