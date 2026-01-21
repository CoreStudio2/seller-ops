import useSWR from 'swr';
import { useSellerOpsStore } from '@/lib/store';
import type { LiveStatus, ThreatEvent } from '@/lib/types';
import { useEffect } from 'react';

const fetcher = (url: string) => fetch(url).then(res => res.json());

export function useRealtimeDashboard() {
    const setLiveStatus = useSellerOpsStore(s => s.setLiveStatus);
    const addThreat = useSellerOpsStore(s => s.addThreat);

    // Poll /api/status every 2 seconds
    const { data, error } = useSWR<{ status: LiveStatus; threats: ThreatEvent[] }>(
        '/api/status',
        fetcher,
        {
            refreshInterval: 5000,
            dedupingInterval: 2000,
        }
    );

    // Update global store when data arrives
    useEffect(() => {
        if (data) {
            if (data.status) setLiveStatus(data.status);

            // Merge threats (simple strategy for demo: just add new ones if we had IDs)
            // For now, let's just reverse the list from API and replace
            // Real app would merge properly

            // Actually, store supports bulk set? No, just addThreat.
            // Let's iterate and add ANY that we don't have? 
            // Or just rely on the API returning the "Feed" and we render the feed from SWR directly?
            // The Store is useful for "Selected Threat" state.

            // Let's create a bulk update action in store or just use SWR data in components.
            // Ideally, the Store holds UI state (selection) and Server State (threats).
            // SWR manages Server State better. 
            // But we built the Store for threats. Let's sync them for now.
        }
    }, [data, setLiveStatus]);

    return {
        status: data?.status,
        threats: data?.threats ?? [],
        isLoading: !data && !error,
        error,
    };
}
