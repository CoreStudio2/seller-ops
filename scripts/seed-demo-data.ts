/**
 * Demo Data Seeding Script
 * Inserts realistic threat events and signals into the system
 * NOTE: This is a placeholder script for future use
 */

import type { Signal, ThreatEvent, SignalType } from '@/lib/types';

// This script is not currently used in the build
// It's kept for reference but doesn't affect production

export async function seedDemoData() {
    console.log('Demo data seeding is currently disabled');
    console.log('Use /api/ingest endpoint to generate demo data');
    return { success: true };
}

if (require.main === module) {
    seedDemoData()
        .then(() => process.exit(0))
        .catch(() => process.exit(1));
}


// Realistic signals
const DEMO_SIGNALS: Omit<Signal, 'id' | 'timestamp'>[] = [
    {
        type: 'COMPETITOR_PRICE' as SignalType,
        value: 1379,
        previousValue: 1499,
        delta: -120
    },
    {
        type: 'CONVERSION_DROP' as SignalType,
        value: 2.1,
        previousValue: 3.8,
        delta: -1.7
    },
    {
        type: 'REFUND_SPIKE' as SignalType,
        value: 23,
        previousValue: 20,
        delta: 3
    },
    {
        type: 'REVENUE_VELOCITY' as SignalType,
        value: -12.5,
        previousValue: 2.3,
        delta: -14.8
    },
    {
        type: 'CART_ABANDONMENT' as SignalType,
        value: 68.5,
        previousValue: 52.3,
        delta: 16.2
    }
];

export async function seedDemoData() {
    console.log('🌱 Seeding demo data...');
    
    try {
        // Publish signals
        for (const signal of DEMO_SIGNALS) {
            const fullSignal: Signal = {
                id: `sig-${Date.now()}-${Math.random().toString(36).substr(2, 9)}` as any,
                timestamp: new Date(Date.now() - Math.random() * 3600000), // Last hour
                ...signal
            };
            
            await publishSignal(fullSignal);
            console.log(`  ✓ Signal published: ${signal.type} (${signal.delta > 0 ? '+' : ''}${signal.delta})`);
        }

        // Publish threats
        for (const threat of DEMO_THREATS) {
            const fullThreat: ThreatEvent = {
                id: `threat-${Date.now()}-${Math.random().toString(36).substr(2, 9)}` as any,
                timestamp: new Date(Date.now() - Math.random() * 1800000), // Last 30 min
                ...threat
            };
            
            await publishThreat(fullThreat);
            console.log(`  ✓ Threat published: ${threat.title} [${threat.severity}]`);
        }

        console.log('✅ Demo data seeded successfully!');
        return { success: true, signals: DEMO_SIGNALS.length, threats: DEMO_THREATS.length };
        
    } catch (error) {
        console.error('❌ Failed to seed demo data:', error);
        throw error;
    }
}

// Run if called directly
if (require.main === module) {
    seedDemoData()
        .then(() => process.exit(0))
        .catch(() => process.exit(1));
}
