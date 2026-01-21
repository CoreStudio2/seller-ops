/**
 * SellerOps Data Service
 * Unified data access layer for War Room intelligence.
 * Abstracts Redis (Hot Storage) and Turso (Cold Storage).
 */

import { publishSignal as publishSignalToRedis, getRecentSignals, updateLiveStatus, getCachedLiveStatus } from './redis/unified-client';
import { saveThreat, getRecentThreats, saveSignal } from './turso/database';
import {
    createSignal,
    detectThreatsFromSignal,
    deserializeSignal,
    serializeSignal,
    generateMockSignal
} from './redis/signals';
import type { LiveStatus, Signal, ThreatEvent, SignalType } from './types';

// === SIGNAL INGESTION ===

/**
 * Ingest a raw signal, checking for threats and persisting to storage.
 * This is the "Brain" of the backend processing.
 */
export async function ingestSignal(type: SignalType, value: number, meta?: any): Promise<{ signal: Signal, threat: ThreatEvent | null }> {
    // 1. Create standardized signal
    const signal = createSignal(type, value, meta);

    // 2. Hot Path: Publish to Redis for real-time feed
    const serialized = serializeSignal(signal);
    await publishSignalToRedis(serialized);

    // 3. Cold Path: Persist to DB for audit
    // (Fire and forget to not block response)
    saveSignal(signal).catch(err => console.error('Failed to audit signal:', err));

    // 4. Intelligence: Detect Threats
    const detection = detectThreatsFromSignal(signal);
    let threat: ThreatEvent | null = null;

    if (detection && detection.shouldAlert) {
        threat = {
            id: `threat-${Date.now()}` as any,
            severity: detection.severity,
            title: `${type.replace(/_/g, ' ')} Detected`,
            description: detection.message,
            signals: [signal.id],
            detectedAt: new Date(),
            confidence: 100, // Deterministic rule = 100% confidence
        };

        // Persist threat
        await saveThreat(threat);

        // Also publish threat to Redis (channel: threats)
        const redis = require('./redis/unified-client');
        await redis.publishSignal(JSON.stringify(threat));
    }

    return { signal, threat };
}

// === LIVE FEED DATA ===

export async function getDashboardData() {
    // 1. Get cached live status (or generate default if missing/stale)
    let liveStatusStr = await getCachedLiveStatus();
    let liveStatus: LiveStatus;

    if (liveStatusStr) {
        liveStatus = JSON.parse(liveStatusStr);
        liveStatus.lastUpdated = new Date(liveStatus.lastUpdated);
    } else {
        // Generate fresh baseline status
        liveStatus = {
            revenueVelocity: 2800 + Math.random() * 400,
            revenueDirection: Math.random() > 0.5 ? 'UP' : 'DOWN',
            riskScore: Math.floor(Math.random() * 30 + 20),
            opportunityScore: Math.floor(Math.random() * 40 + 40),
            activeThreats: 0,
            lastUpdated: new Date()
        };
        await updateLiveStatus(JSON.stringify(liveStatus));
    }

    // 2. Get recent threats from DB (Single Source of Truth)
    let threats = await getRecentThreats(10);

    // SEEDING: If no threats exist, generate a few for the "War Room" experience
    if (threats.length === 0) {
        // Run a few demo cycles to populate DB
        await runDemoCycle();
        await runDemoCycle();
        threats = await getRecentThreats(10);
    }

    liveStatus.activeThreats = threats.filter(t => t.severity === 'CRITICAL' || t.severity === 'WARNING').length;

    return {
        status: liveStatus,
        threats,
    };
}

// === DEMO GENERATOR ===
export async function runDemoCycle() {
    const signal = generateMockSignal();
    return await ingestSignal(signal.type, signal.value, { previousValue: signal.previousValue });
}
