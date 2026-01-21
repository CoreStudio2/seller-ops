/**
 * Demo Data Seeding Script
 * Inserts realistic threat events and signals into the system
 */

import { publishSignal, publishThreat } from '@/lib/redis/signals';
import type { Signal, ThreatEvent, SignalType, SeverityLevel } from '@/lib/types';

// Realistic demo threats
const DEMO_THREATS: Omit<ThreatEvent, 'id' | 'timestamp'>[] = [
    {
        title: 'Competitor Price Attack Detected',
        severity: 'WARNING',
        category: 'PRICE_WAR',
        description: 'Competitor "TechMart" dropped price by 12% on similar products. Risk of losing Buy Box.',
        metrics: {
            impact: 'MEDIUM',
            confidence: 87
        }
    },
    {
        title: 'Sudden Conversion Rate Drop',
        severity: 'CRITICAL',
        category: 'PERFORMANCE_ISSUE',
        description: 'Checkout conversion dropped from 3.8% to 2.1% in last 2 hours. Mobile traffic affected.',
        metrics: {
            impact: 'HIGH',
            confidence: 92
        }
    },
    {
        title: 'Refund Pattern Change',
        severity: 'INFO',
        category: 'QUALITY_ISSUE',
        description: 'Refund rate increased 15% for SKU-8842. Common reason: "Product not as described".',
        metrics: {
            impact: 'LOW',
            confidence: 78
        }
    },
    {
        title: 'High-Value Order Anomaly',
        severity: 'WARNING',
        category: 'FRAUD_RISK',
        description: '3 orders above ₹50,000 from new accounts in last hour. Unusual shipping addresses.',
        metrics: {
            impact: 'MEDIUM',
            confidence: 81
        }
    },
    {
        title: 'Ad Campaign Performance Drop',
        severity: 'WARNING',
        category: 'MARKETING',
        description: 'Campaign ROI dropped to 0.7x. Spending ₹12,000/day with negative returns.',
        metrics: {
            impact: 'MEDIUM',
            confidence: 95
        }
    }
];

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
