import { NextRequest, NextResponse } from 'next/server';
import { z } from 'zod';
import { runSimulation } from '@/lib/simulation/engine';

// Request validation schema
const SimulationParamsSchema = z.object({
    priceChange: z.number().min(-100).max(100),
    adSpendChange: z.number().min(-100).max(200),
    shippingSpeedChange: z.number().min(-7).max(7),
});

export async function POST(request: NextRequest) {
    try {
        const body = await request.json();

        // Validate input
        const params = SimulationParamsSchema.parse(body);

        // Run simulation
        const result = runSimulation(params);

        return NextResponse.json(result);
    } catch (error) {
        if (error instanceof z.ZodError) {
            return NextResponse.json(
                { error: 'Invalid parameters', details: error.issues },
                { status: 400 }
            );
        }

        console.error('Simulation error:', error);
        return NextResponse.json(
            { error: 'Simulation failed' },
            { status: 500 }
        );
    }
}
