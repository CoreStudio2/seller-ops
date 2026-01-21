import { NextResponse } from 'next/server';
import { initializeDatabase } from '@/lib/turso/database';

export async function GET() {
    try {
        const db = await initializeDatabase();
        return NextResponse.json({ status: 'ok', message: 'Database initialized' });
    } catch (error) {
        console.error('DB Init failed:', error);
        return NextResponse.json({ error: 'Failed to init DB' }, { status: 500 });
    }
}
