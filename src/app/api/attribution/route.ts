import { NextRequest, NextResponse } from 'next/server';
import { generateAttributionBrief } from '@/lib/gemini/attribution';
import type { ThreatEvent, Signal, ThreatId, SignalId } from '@/lib/types';

// Demo signals for testing without real data
const DEMO_SIGNALS: Signal[] = [
    {
        id: 'sig-1' as SignalId,
        type: 'COMPETITOR_PRICE',
        timestamp: new Date(Date.now() - 3600000),
        value: 1379,
        previousValue: 1499,
        delta: -120,
    },
    {
        id: 'sig-2' as SignalId,
        type: 'CONVERSION_DROP',
        timestamp: new Date(Date.now() - 1800000),
        value: 2.1,
        previousValue: 3.8,
        delta: -1.7,
    },
];

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();
        const threat = body.threat as ThreatEvent;

        if (!threat || !threat.id || !threat.title) {
            return NextResponse.json(
                { error: 'Invalid threat data' },
                { status: 400 }
            );
        }

        // Use demo signals or provided signals
        const signals = body.signals ?? DEMO_SIGNALS;

        // Generate attribution brief using Gemini
        const brief = await generateAttributionBrief(threat, signals);

        return NextResponse.json(brief);
    } catch (error) {
        console.error('Attribution error:', error);

        // Check if it's an API key error
        if (error instanceof Error && error.message.includes('API')) {
            return NextResponse.json(
                { error: 'Gemini API configuration error', details: error.message },
                { status: 503 }
            );
        }

        return NextResponse.json(
            { error: 'Attribution generation failed' },
            { status: 500 }
        );
    }
}
