'use client';

import { useState, useCallback } from 'react';
import { useSimulationResult, useSellerOpsStore } from '@/lib/store';
import { quickPriceImpact } from '@/lib/simulation/engine';
import type { SimulationParams, SimulationResult, SeverityLevel } from '@/lib/types';

// === SLIDER COMPONENT ===
interface SliderProps {
    label: string;
    value: number;
    min: number;
    max: number;
    step?: number;
    unit?: string;
    onChange: (value: number) => void;
}

function Slider({ label, value, min, max, step = 1, unit = '%', onChange }: SliderProps) {
    const percentage = ((value - min) / (max - min)) * 100;

    return (
        <div className="space-y-2">
            <div className="flex justify-between items-center">
                <span className="font-mono text-xs uppercase tracking-wider text-text-tertiary">
                    {label}
                </span>
                <span className={`font-mono text-sm font-bold ${value > 0 ? 'text-signal-green' : value < 0 ? 'text-signal-red' : 'text-text-primary'
                    }`}>
                    {value > 0 ? '+' : ''}{value}{unit}
                </span>
            </div>
            <div className="relative h-2 bg-surface-overlay">
                <div
                    className={`absolute h-full ${value >= 0 ? 'bg-signal-green' : 'bg-signal-red'}`}
                    style={{
                        width: `${Math.abs(percentage - 50) * 2}%`,
                        left: value >= 0 ? '50%' : `${percentage}%`,
                        right: value >= 0 ? `${100 - percentage}%` : '50%',
                    }}
                />
                <div
                    className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-text-primary border-2 border-surface-base"
                    style={{ left: `calc(${percentage}% - 8px)` }}
                />
                <input
                    type="range"
                    min={min}
                    max={max}
                    step={step}
                    value={value}
                    onChange={(e) => onChange(Number(e.target.value))}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
            </div>
        </div>
    );
}

// === RESULT CARD ===
interface ResultCardProps {
    label: string;
    value: string | number;
    subtext?: string;
    severity?: SeverityLevel;
}

function ResultCard({ label, value, subtext, severity = 'INFO' }: ResultCardProps) {
    const getSeverityClass = () => {
        switch (severity) {
            case 'CRITICAL': return 'border-signal-red text-signal-red';
            case 'WARNING': return 'border-signal-amber text-signal-amber';
            case 'OK': return 'border-signal-green text-signal-green';
            default: return 'border-signal-cyan text-signal-cyan';
        }
    };

    return (
        <div className={`hud-panel p-4 border-l-4 ${getSeverityClass()}`}>
            <span className="font-mono text-xs uppercase tracking-wider text-text-tertiary block mb-1">
                {label}
            </span>
            <span className="font-mono text-2xl font-bold block">
                {value}
            </span>
            {subtext && (
                <span className="font-mono text-xs text-text-secondary">
                    {subtext}
                </span>
            )}
        </div>
    );
}

// === BEAST MODE PANEL ===
export function BeastModePanel() {
    const [params, setParams] = useState<SimulationParams>({
        priceChange: 0,
        adSpendChange: 0,
        shippingSpeedChange: 0,
    });

    const [localResult, setLocalResult] = useState<{
        revenueChange: string;
        marginChange: string;
        competitorRisk: string;
    } | null>(null);

    const simulationResult = useSimulationResult();
    const runSimulation = useSellerOpsStore((s) => s.runSimulation);
    const isSimulating = useSellerOpsStore((s) => s.isSimulating);

    // Real-time preview as sliders move
    const handleParamChange = useCallback((
        key: keyof SimulationParams,
        value: number
    ) => {
        const newParams = { ...params, [key]: value };
        setParams(newParams);

        // Quick local calculation for instant feedback
        const quick = quickPriceImpact(newParams.priceChange);
        setLocalResult(quick);
    }, [params]);

    // Full simulation on button click
    const handleSimulate = useCallback(async () => {
        await runSimulation(params);
    }, [params, runSimulation]);

    const displayResult = simulationResult ?? {
        projectedRevenue: 50000,
        projectedMargin: 22,
        projectedVolume: 500,
        competitorResponseProbability: 0,
        riskLevel: 'OK' as SeverityLevel,
    };

    return (
        <div className="flex-1 p-6 bg-surface-base">
            {/* Header */}
            <div className="flex items-center gap-3 mb-6">
                <div className="w-10 h-10 bg-signal-amber flex items-center justify-center">
                    <span className="font-mono font-bold text-black text-xl">⚔</span>
                </div>
                <div>
                    <h2 className="font-mono text-xl font-bold text-text-primary">
                        BEAST MODE
                    </h2>
                    <p className="font-mono text-xs text-text-tertiary uppercase tracking-wider">
                        Simulation & Counter-Strategy
                    </p>
                </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
                {/* Left: Controls */}
                <div className="space-y-6">
                    <div className="hud-panel p-4">
                        <h3 className="font-mono text-sm font-bold text-text-primary mb-4 uppercase tracking-wider">
                            Adjust Parameters
                        </h3>

                        <div className="space-y-6">
                            <Slider
                                label="Price Change"
                                value={params.priceChange}
                                min={-20}
                                max={20}
                                onChange={(v) => handleParamChange('priceChange', v)}
                            />

                            <Slider
                                label="Ad Spend Change"
                                value={params.adSpendChange}
                                min={-50}
                                max={100}
                                step={5}
                                onChange={(v) => handleParamChange('adSpendChange', v)}
                            />

                            <Slider
                                label="Shipping Speed"
                                value={params.shippingSpeedChange}
                                min={-3}
                                max={3}
                                unit=" days"
                                onChange={(v) => handleParamChange('shippingSpeedChange', v)}
                            />
                        </div>

                        <button
                            onClick={handleSimulate}
                            disabled={isSimulating}
                            className={`
                w-full mt-6 py-3 px-4 font-mono text-sm font-bold uppercase tracking-wider
                transition-all duration-200
                ${isSimulating
                                    ? 'bg-surface-overlay text-text-tertiary cursor-wait'
                                    : 'bg-signal-amber text-black hover:bg-signal-green'
                                }
              `}
                        >
                            {isSimulating ? 'Simulating...' : 'Run Full Simulation'}
                        </button>
                    </div>

                    {/* Quick Preview */}
                    {localResult && (
                        <div className="hud-panel p-4">
                            <h3 className="font-mono text-xs text-text-tertiary uppercase tracking-wider mb-3">
                                Quick Preview
                            </h3>
                            <div className="grid grid-cols-3 gap-2 text-center">
                                <div>
                                    <span className="font-mono text-lg font-bold text-signal-cyan block">
                                        {localResult.revenueChange}
                                    </span>
                                    <span className="font-mono text-xs text-text-tertiary">Revenue</span>
                                </div>
                                <div>
                                    <span className="font-mono text-lg font-bold text-signal-amber block">
                                        {localResult.marginChange}
                                    </span>
                                    <span className="font-mono text-xs text-text-tertiary">Margin</span>
                                </div>
                                <div>
                                    <span className="font-mono text-lg font-bold text-signal-red block">
                                        {localResult.competitorRisk}
                                    </span>
                                    <span className="font-mono text-xs text-text-tertiary">Comp. Risk</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right: Results */}
                <div className="space-y-4">
                    <h3 className="font-mono text-sm font-bold text-text-primary uppercase tracking-wider">
                        Projected Outcomes
                    </h3>

                    <div className="grid grid-cols-2 gap-4">
                        <ResultCard
                            label="Projected Revenue"
                            value={`₹${displayResult.projectedRevenue.toLocaleString()}`}
                            subtext="per day"
                            severity={displayResult.projectedRevenue > 50000 ? 'OK' : 'WARNING'}
                        />
                        <ResultCard
                            label="Projected Margin"
                            value={`${displayResult.projectedMargin}%`}
                            subtext="net margin"
                            severity={displayResult.projectedMargin > 20 ? 'OK' : displayResult.projectedMargin > 10 ? 'WARNING' : 'CRITICAL'}
                        />
                        <ResultCard
                            label="Volume"
                            value={displayResult.projectedVolume}
                            subtext="units/day"
                            severity="INFO"
                        />
                        <ResultCard
                            label="Competitor Response"
                            value={`${displayResult.competitorResponseProbability}%`}
                            subtext="probability"
                            severity={
                                displayResult.competitorResponseProbability > 60 ? 'CRITICAL' :
                                    displayResult.competitorResponseProbability > 30 ? 'WARNING' : 'OK'
                            }
                        />
                    </div>

                    {/* Risk Assessment */}
                    <div className={`
            hud-panel p-4 mt-4
            ${displayResult.riskLevel === 'CRITICAL' ? 'border-signal-red' :
                            displayResult.riskLevel === 'WARNING' ? 'border-signal-amber' :
                                'border-signal-green'}
          `}>
                        <div className="flex items-center justify-between">
                            <span className="font-mono text-sm text-text-tertiary">Overall Risk</span>
                            <span className={`
                font-mono text-lg font-bold px-3 py-1
                ${displayResult.riskLevel === 'CRITICAL' ? 'signal-critical' :
                                    displayResult.riskLevel === 'WARNING' ? 'signal-warning' :
                                        'signal-ok'}
              `}>
                                {displayResult.riskLevel}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
