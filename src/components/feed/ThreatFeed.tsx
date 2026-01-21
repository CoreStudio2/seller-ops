'use client';

import { useThreats, useSelectedThreat, useSellerOpsStore } from '@/lib/store';
import type { ThreatEvent, SeverityLevel } from '@/lib/types';

// === SEVERITY BADGE ===
function SeverityBadge({ severity }: { severity: SeverityLevel }) {
    const getClasses = () => {
        switch (severity) {
            case 'CRITICAL':
                return 'signal-critical animate-pulse-glow';
            case 'WARNING':
                return 'signal-warning';
            case 'INFO':
                return 'signal-info';
            case 'OK':
                return 'signal-ok';
        }
    };

    return (
        <span className={`px-2 py-0.5 text-xs font-mono font-bold uppercase ${getClasses()}`}>
            {severity}
        </span>
    );
}

// === THREAT CARD ===
interface ThreatCardProps {
    threat: ThreatEvent;
    isSelected: boolean;
    onClick: () => void;
}

// Helper function to calculate time since (moved outside component to avoid render impurity)
function formatTimeSince(date: Date, now: number): string {
    const seconds = Math.floor((now - date.getTime()) / 1000);
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
}

function ThreatCard({ threat, isSelected, onClick }: ThreatCardProps) {
    // Get current time once per render cycle (acceptable side effect)
    const now = Date.now();

    return (
        <button
            onClick={onClick}
            className={`
        w-full text-left p-3 hud-panel transition-all duration-200
        ${isSelected
                    ? 'border-signal-green hud-panel-glow'
                    : 'hover:border-text-tertiary'
                }
        animate-slide-in
      `}
        >
            <div className="flex items-start justify-between gap-2 mb-2">
                <SeverityBadge severity={threat.severity} />
                <span className="font-mono text-xs text-text-tertiary">
                    {formatTimeSince(threat.detectedAt, now)}
                </span>
            </div>

            <h3 className="font-mono text-sm font-bold text-text-primary mb-1 leading-tight">
                {threat.title}
            </h3>

            <p className="font-mono text-xs text-text-secondary line-clamp-2">
                {threat.description}
            </p>

            {threat.suggestedAction && (
                <div className="mt-2 pt-2 border-t border-border-default">
                    <span className="font-mono text-xs text-signal-cyan">
                        → {threat.suggestedAction}
                    </span>
                </div>
            )}

            <div className="mt-2 flex items-center justify-between">
                <span className="font-mono text-xs text-text-tertiary">
                    Confidence: {threat.confidence}%
                </span>
                <span className="font-mono text-xs text-text-tertiary">
                    {threat.signals.length} signals
                </span>
            </div>
        </button>
    );
}

// === THREAT FEED ===
export function ThreatFeed({ onClose }: { onClose?: () => void }) {
    const threats = useThreats();
    const selectedThreat = useSelectedThreat();
    const selectThreat = useSellerOpsStore((s) => s.selectThreat);

    const displayThreats = threats; // Use ONLY real threats from store

    return (
        <aside className="w-full md:w-80 h-full bg-surface-base border-r border-border-default flex flex-col">
            {/* Header */}
            <div className="p-3 border-b border-border-default">
                <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 bg-signal-red animate-pulse-glow" />
                        <h2 className="font-mono text-sm font-bold uppercase tracking-wider text-text-primary">
                            Threat Feed
                        </h2>
                    </div>
                    {/* Close button for mobile */}
                    {onClose && (
                        <button
                            onClick={onClose}
                            className="md:hidden text-text-secondary hover:text-text-primary text-xl"
                            aria-label="Close threat feed"
                        >
                            ✕
                        </button>
                    )}
                </div>
                <p className="font-mono text-xs text-text-tertiary mt-1">
                    {displayThreats.length} active • Click to analyze
                </p>
            </div>

            {/* Threat List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-2">
                {displayThreats.map((threat) => (
                    <ThreatCard
                        key={threat.id}
                        threat={threat}
                        isSelected={selectedThreat?.id === threat.id}
                        onClick={() => selectThreat(threat)}
                    />
                ))}
            </div>

            {/* Footer */}
            <div className="p-3 border-t border-border-default">
                <button className="w-full py-2 px-3 font-mono text-xs uppercase tracking-wider text-text-secondary hover:text-signal-green transition-colors border border-border-default hover:border-signal-green">
                    View All History →
                </button>
            </div>
        </aside>
    );
}
