'use client';

import { useLiveStatus } from '@/lib/store';
import type { LiveStatus, SeverityLevel } from '@/lib/types';

// === METRIC CARD COMPONENT ===
interface MetricCardProps {
    label: string;
    value: string | number;
    direction?: 'UP' | 'DOWN' | 'STABLE';
    severity?: SeverityLevel;
}

function MetricCard({ label, value, direction, severity }: MetricCardProps) {
    const getDirectionIcon = () => {
        switch (direction) {
            case 'UP': return '↑';
            case 'DOWN': return '↓';
            default: return '→';
        }
    };

    const getSeverityClass = () => {
        switch (severity) {
            case 'CRITICAL': return 'text-signal-red';
            case 'WARNING': return 'text-signal-amber';
            case 'OK': return 'text-signal-green';
            default: return 'text-signal-cyan';
        }
    };

    const getDirectionClass = () => {
        switch (direction) {
            case 'UP': return 'text-signal-green';
            case 'DOWN': return 'text-signal-red';
            default: return 'text-text-secondary';
        }
    };

    return (
        <div className="hud-panel px-4 py-3 flex flex-col gap-1">
            <span className="text-xs uppercase tracking-wider text-text-tertiary font-mono">
                {label}
            </span>
            <div className="flex items-baseline gap-2">
                <span className={`text-2xl font-bold font-mono ${getSeverityClass()}`}>
                    {value}
                </span>
                {direction && (
                    <span className={`text-lg font-mono ${getDirectionClass()}`}>
                        {getDirectionIcon()}
                    </span>
                )}
            </div>
        </div>
    );
}

// === THREAT INDICATOR ===
function ThreatIndicator({ count, severity }: { count: number; severity: SeverityLevel }) {
    const getIndicatorClass = () => {
        if (count === 0) return 'bg-signal-green';
        switch (severity) {
            case 'CRITICAL': return 'bg-signal-red animate-pulse-glow';
            case 'WARNING': return 'bg-signal-amber';
            default: return 'bg-signal-cyan';
        }
    };

    return (
        <div className="hud-panel px-4 py-3 flex items-center gap-3">
            <div className={`w-3 h-3 ${getIndicatorClass()}`} />
            <div className="flex flex-col">
                <span className="text-xs uppercase tracking-wider text-text-tertiary font-mono">
                    Active Threats
                </span>
                <span className={`text-2xl font-bold font-mono ${count > 0 ? 'text-signal-red' : 'text-signal-green'
                    }`}>
                    {count}
                </span>
            </div>
        </div>
    );
}

// === MAIN STATUS BAR ===
export function LiveStatusBar() {
    const status = useLiveStatus();

    if (!status) {
        return (
            <header className="w-full bg-surface-base border-b border-border-default h-20 flex items-center justify-between px-4 animate-pulse">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-surface-elevated" />
                    <div className="h-8 w-32 bg-surface-elevated" />
                </div>
                <div className="flex gap-2">
                    {[1, 2, 3, 4].map(i => (
                        <div key={i} className="h-14 w-32 bg-surface-elevated" />
                    ))}
                </div>
                <div className="h-8 w-24 bg-surface-elevated" />
            </header>
        );
    }

    const getRiskSeverity = (score: number): SeverityLevel => {
        if (score >= 70) return 'CRITICAL';
        if (score >= 40) return 'WARNING';
        return 'OK';
    };

    return (
        <header className="w-full bg-surface-base border-b border-border-default">
            <div className="flex items-center justify-between px-4 py-2">
                {/* Logo / Title */}
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-signal-green flex items-center justify-center">
                        <span className="font-mono font-bold text-black">SO</span>
                    </div>
                    <div>
                        <h1 className="font-mono text-lg font-bold tracking-tight text-text-primary">
                            SELLEROPS
                        </h1>
                        <span className="font-mono text-xs text-text-tertiary uppercase tracking-widest">
                            War Room Active
                        </span>
                    </div>
                </div>

                {/* Metrics Grid */}
                <div className="flex gap-2">
                    <MetricCard
                        label="Revenue Velocity"
                        value={`₹${status.revenueVelocity.toLocaleString()}/hr`}
                        direction={status.revenueDirection}
                    />
                    <MetricCard
                        label="Risk Score"
                        value={status.riskScore}
                        severity={getRiskSeverity(status.riskScore)}
                    />
                    <MetricCard
                        label="Opportunity"
                        value={status.opportunityScore}
                        severity={status.opportunityScore > 50 ? 'OK' : 'INFO'}
                    />
                    <ThreatIndicator
                        count={status.activeThreats}
                        severity={status.activeThreats >= 1 ? (status.activeThreats >= 3 ? 'CRITICAL' : 'WARNING') : 'OK'}
                    />
                </div>

                {/* Timestamp */}
                <div className="text-right">
                    <span className="font-mono text-xs text-text-tertiary block">
                        Last Update
                    </span>
                    <span className="font-mono text-sm text-text-secondary" suppressHydrationWarning>
                        {new Date(status.lastUpdated).toLocaleTimeString()}
                    </span>
                </div>
            </div>
        </header>
    );
}
