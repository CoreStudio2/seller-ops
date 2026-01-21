/**
 * Turso Database Client
 * Edge SQLite for threat memory and signal auditing
 */

import { createClient } from '@libsql/client';
import type { ThreatEvent, Signal, ThreatId, SignalId } from '@/lib/types';

// === DATABASE SCHEMA ===
export const SCHEMA = `
-- Threat Events Table
CREATE TABLE IF NOT EXISTS threat_events (
  id TEXT PRIMARY KEY,
  severity TEXT NOT NULL CHECK (severity IN ('CRITICAL', 'WARNING', 'INFO', 'OK')),
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  suggested_action TEXT,
  confidence INTEGER NOT NULL CHECK (confidence >= 0 AND confidence <= 100),
  detected_at TEXT NOT NULL,
  resolved_at TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Signal Audit Table
CREATE TABLE IF NOT EXISTS signal_audit (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  value REAL NOT NULL,
  previous_value REAL,
  delta REAL,
  sku_id TEXT,
  metadata TEXT,
  timestamp TEXT NOT NULL,
  threat_id TEXT REFERENCES threat_events(id),
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Threat-Signal Junction Table
CREATE TABLE IF NOT EXISTS threat_signals (
  threat_id TEXT NOT NULL REFERENCES threat_events(id),
  signal_id TEXT NOT NULL REFERENCES signal_audit(id),
  PRIMARY KEY (threat_id, signal_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_threat_severity ON threat_events(severity);
CREATE INDEX IF NOT EXISTS idx_threat_detected_at ON threat_events(detected_at);
CREATE INDEX IF NOT EXISTS idx_signal_type ON signal_audit(type);
CREATE INDEX IF NOT EXISTS idx_signal_timestamp ON signal_audit(timestamp);
`;

// === CLIENT FACTORY ===
let dbClient: ReturnType<typeof createClient> | null = null;

export function getDatabase() {
    if (!dbClient) {
        const url = process.env.TURSO_DATABASE_URL;
        const authToken = process.env.TURSO_AUTH_TOKEN;

        if (!url) {
            // Use local SQLite for development
            dbClient = createClient({
                url: 'file:local.db',
            });
        } else {
            dbClient = createClient({
                url,
                authToken,
            });
        }
    }
    return dbClient;
}

// === INITIALIZE DATABASE ===
export async function initializeDatabase() {
    const db = getDatabase();
    await db.executeMultiple(SCHEMA);
    return db;
}

// === THREAT OPERATIONS ===

export async function saveThreat(threat: ThreatEvent): Promise<void> {
    const db = getDatabase();

    await db.execute({
        sql: `
      INSERT INTO threat_events (id, severity, title, description, suggested_action, confidence, detected_at)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `,
        args: [
            threat.id,
            threat.severity,
            threat.title,
            threat.description,
            threat.suggestedAction ?? null,
            threat.confidence,
            threat.detectedAt.toISOString(),
        ],
    });

    // Link signals if any
    for (const signalId of threat.signals) {
        await db.execute({
            sql: `INSERT OR IGNORE INTO threat_signals (threat_id, signal_id) VALUES (?, ?)`,
            args: [threat.id, signalId],
        });
    }
}

export async function getThreat(id: ThreatId): Promise<ThreatEvent | null> {
    const db = getDatabase();

    const result = await db.execute({
        sql: `SELECT * FROM threat_events WHERE id = ?`,
        args: [id],
    });

    if (result.rows.length === 0) return null;

    const row = result.rows[0];

    // Get linked signals
    const signalsResult = await db.execute({
        sql: `SELECT signal_id FROM threat_signals WHERE threat_id = ?`,
        args: [id],
    });

    return {
        id: row.id as ThreatId,
        severity: row.severity as ThreatEvent['severity'],
        title: row.title as string,
        description: row.description as string,
        suggestedAction: row.suggested_action as string | undefined,
        confidence: row.confidence as number,
        detectedAt: new Date(row.detected_at as string),
        signals: signalsResult.rows.map(r => r.signal_id as SignalId),
    };
}

export async function getRecentThreats(
    limit: number = 50,
    severity?: ThreatEvent['severity']
): Promise<ThreatEvent[]> {
    const db = getDatabase();

    let sql = `SELECT * FROM threat_events`;
    const args: (string | number)[] = [];

    if (severity) {
        sql += ` WHERE severity = ?`;
        args.push(severity);
    }

    sql += ` ORDER BY detected_at DESC LIMIT ?`;
    args.push(limit);

    const result = await db.execute({ sql, args });

    return result.rows.map(row => ({
        id: row.id as ThreatId,
        severity: row.severity as ThreatEvent['severity'],
        title: row.title as string,
        description: row.description as string,
        suggestedAction: row.suggested_action as string | undefined,
        confidence: row.confidence as number,
        detectedAt: new Date(row.detected_at as string),
        signals: [], // Lazy load if needed
    }));
}

// === SIGNAL OPERATIONS ===

export async function saveSignal(signal: Signal, threatId?: ThreatId): Promise<void> {
    const db = getDatabase();

    await db.execute({
        sql: `
      INSERT INTO signal_audit (id, type, value, previous_value, delta, sku_id, metadata, timestamp, threat_id)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `,
        args: [
            signal.id,
            signal.type,
            signal.value,
            signal.previousValue ?? null,
            signal.delta ?? null,
            signal.skuId ?? null,
            signal.metadata ? JSON.stringify(signal.metadata) : null,
            signal.timestamp.toISOString(),
            threatId ?? null,
        ],
    });
}

export async function getSignalsForThreat(threatId: ThreatId): Promise<Signal[]> {
    const db = getDatabase();

    const result = await db.execute({
        sql: `
      SELECT s.* FROM signal_audit s
      JOIN threat_signals ts ON s.id = ts.signal_id
      WHERE ts.threat_id = ?
      ORDER BY s.timestamp DESC
    `,
        args: [threatId],
    });

    return result.rows.map(row => ({
        id: row.id as SignalId,
        type: row.type as Signal['type'],
        value: row.value as number,
        previousValue: row.previous_value as number | undefined,
        delta: row.delta as number | undefined,
        skuId: row.sku_id as Signal['skuId'],
        metadata: row.metadata ? JSON.parse(row.metadata as string) : undefined,
        timestamp: new Date(row.timestamp as string),
    }));
}

// === ANALYTICS QUERIES ===

export async function getThreatCountsBySevenity(): Promise<Record<string, number>> {
    const db = getDatabase();

    const result = await db.execute({
        sql: `
      SELECT severity, COUNT(*) as count
      FROM threat_events
      WHERE detected_at > datetime('now', '-7 days')
      GROUP BY severity
    `,
        args: [],
    });

    const counts: Record<string, number> = {};
    for (const row of result.rows) {
        counts[row.severity as string] = row.count as number;
    }
    return counts;
}

export async function getSignalTypeDistribution(): Promise<Record<string, number>> {
    const db = getDatabase();

    const result = await db.execute({
        sql: `
      SELECT type, COUNT(*) as count
      FROM signal_audit
      WHERE timestamp > datetime('now', '-24 hours')
      GROUP BY type
      ORDER BY count DESC
    `,
        args: [],
    });

    const distribution: Record<string, number> = {};
    for (const row of result.rows) {
        distribution[row.type as string] = row.count as number;
    }
    return distribution;
}
