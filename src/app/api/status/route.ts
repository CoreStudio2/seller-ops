import { NextResponse } from 'next/server';
import { getDashboardData } from '@/lib/data';

export const dynamic = 'force-dynamic'; // Always real-time

export async function GET() {
    try {
        const data = await getDashboardData();
        return NextResponse.json(data);
    } catch (error) {
        console.error('Failed to fetching dashboard status:', error);
        return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
    }
}
