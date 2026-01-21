import { NextResponse } from 'next/server';
import { runDemoCycle } from '@/lib/data';

export async function POST() {
    try {
        const result = await runDemoCycle();
        return NextResponse.json(result);
    } catch (error) {
        console.error('Ingest failed:', error);
        return NextResponse.json({ error: 'Ingest Failed' }, { status: 500 });
    }
}
