'use client';

import { useState, useEffect } from 'react';
import { useSelectedThreat, useAttributionBrief, useSellerOpsStore } from '@/lib/store';
import type { AttributionBrief, CausalFactor, SuggestedAction } from '@/lib/types';

// === TYPING EFFECT HOOK ===
function useTypingEffect(text: string, speed: number = 30) {
    const [displayedText, setDisplayedText] = useState('');
    const [isComplete, setIsComplete] = useState(false);

    useEffect(() => {
        if (!text) {
            setDisplayedText('');
            setIsComplete(true);
            return;
        }

        setDisplayedText('');
        setIsComplete(false);
        let index = 0;

        const timer = setInterval(() => {
            if (index < text.length) {
                setDisplayedText((prev) => prev + text[index]);
                index++;
            } else {
                setIsComplete(true);
                clearInterval(timer);
            }
        }, speed);

        return () => clearInterval(timer);
    }, [text, speed]);

    return { displayedText, isComplete };
}

// === CAUSAL FACTOR CARD ===
function CausalFactorCard({ factor, index }: { factor: CausalFactor; index: number }) {
    const getImpactColor = () => {
        switch (factor.impact) {
            case 'HIGH': return 'text-signal-red';
            case 'MEDIUM': return 'text-signal-amber';
            case 'LOW': return 'text-signal-cyan';
        }
    };

    return (
        <div
            className="hud-panel p-3 animate-slide-in"
            style={{ animationDelay: `${index * 100}ms` }}
        >
            <div className="flex items-start gap-3">
                <span className={`font-mono text-lg font-bold ${getImpactColor()}`}>
                    {index + 1}
                </span>
                <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                        <span className="font-mono text-sm font-bold text-text-primary">
                            {factor.factor}
                        </span>
                        <span className={`px-2 py-0.5 text-xs font-mono uppercase ${getImpactColor()} bg-surface-overlay`}>
                            {factor.impact}
                        </span>
                    </div>
                    <p className="font-mono text-xs text-text-secondary">
                        {factor.evidence}
                    </p>
                </div>
            </div>
        </div>
    );
}

// === ACTION CARD ===
function ActionCard({ action, index }: { action: SuggestedAction; index: number }) {
    const getPriorityColor = () => {
        switch (action.priority) {
            case 'IMMEDIATE': return 'bg-signal-red text-white';
            case 'SOON': return 'bg-signal-amber text-black';
            case 'OPTIONAL': return 'bg-signal-cyan text-black';
        }
    };

    return (
        <div
            className="hud-panel p-3 animate-slide-in border-l-4 border-signal-green"
            style={{ animationDelay: `${(index + 3) * 100}ms` }}
        >
            <div className="flex items-start justify-between gap-3 mb-2">
                <span className="font-mono text-sm font-bold text-text-primary">
                    → {action.action}
                </span>
                <span className={`px-2 py-0.5 text-xs font-mono font-bold ${getPriorityColor()}`}>
                    {action.priority}
                </span>
            </div>
            <p className="font-mono text-xs text-text-secondary">
                Expected: {action.expectedOutcome}
            </p>
        </div>
    );
}

// === ATTRIBUTION BRIEF PANEL ===
export function AttributionBriefPanel() {
    const selectedThreat = useSelectedThreat();
    const brief = useAttributionBrief();
    const isLoadingBrief = useSellerOpsStore((s) => s.isLoadingBrief);
    const setAttributionBrief = useSellerOpsStore((s) => s.setAttributionBrief);

    const [error, setError] = useState<string | null>(null);

    // Load brief when threat changes
    useEffect(() => {
        const fetchBrief = async () => {
            if (selectedThreat) {
                // If we already have a brief *for this threat*, don't refetch
                if (brief?.threatId === selectedThreat.id) return;

                setError(null);
                setAttributionBrief(null); // Clear previous

                try {
                    const response = await fetch('/api/attribution', {
                        method: 'POST',
                        body: JSON.stringify({ threat: selectedThreat }),
                    });

                    if (response.ok) {
                        const realBrief = await response.json();
                        setAttributionBrief(realBrief);
                    } else {
                        setError('Attribution server unavailable.');
                    }
                } catch (err) {
                    setError('Failed to connect to Intelligence Core.');
                }
            }
        };

        fetchBrief();
    }, [selectedThreat?.id]);

    const displayBrief = brief;
    const { displayedText, isComplete } = useTypingEffect(
        error || (displayBrief?.summary ?? (isLoadingBrief ? 'Analyzing signals...' : '')),
        20
    );

    if (!selectedThreat) {
        return (
            <div className="flex-1 flex items-center justify-center p-8 bg-surface-base">
                <div className="text-center">
                    <div className="w-16 h-16 mx-auto mb-4 border-2 border-border-default flex items-center justify-center">
                        <span className="font-mono text-2xl text-text-tertiary">◎</span>
                    </div>
                    <h3 className="font-mono text-lg text-text-secondary mb-2">
                        No Threat Selected
                    </h3>
                    <p className="font-mono text-sm text-text-tertiary">
                        Select a threat from the feed to view its attribution analysis
                    </p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex-1 p-6 bg-surface-base overflow-y-auto">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-signal-green flex items-center justify-center">
                    <span className="font-mono font-bold text-black text-xl">AI</span>
                </div>
                <div>
                    <h2 className="font-mono text-xl font-bold text-text-primary">
                        ATTRIBUTION BRIEF
                    </h2>
                    <p className="font-mono text-xs text-text-tertiary uppercase tracking-wider">
                        AI Analysis • Confidence {displayBrief?.confidence ?? 0}%
                    </p>
                </div>
            </div>

            {/* Summary with typing effect */}
            <div className="hud-panel p-4 mb-6">
                <h3 className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-2">
                    Executive Summary
                </h3>
                <p className="font-mono text-sm text-text-primary leading-relaxed">
                    {displayedText}
                    {!isComplete && <span className="animate-terminal-blink">▌</span>}
                </p>
            </div>

            {isComplete && displayBrief && (
                <>
                    {/* Causal Factors */}
                    <div className="mb-6">
                        <h3 className="font-mono text-sm font-bold text-text-primary uppercase tracking-wider mb-3">
                            Root Causes
                        </h3>
                        <div className="space-y-2">
                            {displayBrief.causes.map((factor, i) => (
                                <CausalFactorCard key={i} factor={factor} index={i} />
                            ))}
                        </div>
                    </div>

                    {/* Suggested Actions */}
                    <div>
                        <h3 className="font-mono text-sm font-bold text-text-primary uppercase tracking-wider mb-3">
                            Recommended Actions
                        </h3>
                        <div className="space-y-2">
                            {displayBrief.suggestedActions.map((action, i) => (
                                <ActionCard key={i} action={action} index={i} />
                            ))}
                        </div>
                    </div>

                    {/* Action Button */}
                    <div className="mt-6 flex gap-3">
                        <button className="flex-1 py-3 px-4 bg-signal-green text-black font-mono text-sm font-bold uppercase tracking-wider hover:bg-signal-amber transition-colors">
                            Execute Top Action
                        </button>
                        <button className="py-3 px-4 border border-border-default font-mono text-sm text-text-secondary hover:text-signal-cyan hover:border-signal-cyan transition-colors">
                            Simulate First
                        </button>
                    </div>
                </>
            )}
        </div>
    );
}
