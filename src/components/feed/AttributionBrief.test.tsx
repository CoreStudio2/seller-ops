import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { AttributionBriefPanel } from './AttributionBrief';
import type { ThreatEvent, ThreatId, SignalId } from '@/lib/types';

// Mock dependencies
const mockSetBrief = vi.fn();
const mockBrief = null;
let mockSelectedThreat: ThreatEvent | null = null;

vi.mock('@/lib/store', () => ({
    useSelectedThreat: vi.fn(() => mockSelectedThreat),
    useAttributionBrief: vi.fn(() => mockBrief),
    useSellerOpsStore: vi.fn((selector) => selector({
        isLoadingBrief: false,
        setAttributionBrief: mockSetBrief
    })),
}));

// Mock fetch
global.fetch = vi.fn();

describe('AttributionBriefPanel', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockSelectedThreat = {
            id: 'threat-1' as ThreatId,
            severity: 'CRITICAL',
            title: 'Loss of Buy Box',
            description: 'Price war detected',
            signals: ['sig-1' as SignalId],
            detectedAt: new Date(),
            confidence: 90,
        };
    });

    it('should render empty state when no threat selected', () => {
        mockSelectedThreat = null;
        render(<AttributionBriefPanel />);
        expect(screen.getByText('No Threat Selected')).toBeInTheDocument();
    });

    it('should render header when threat selected', () => {
        render(<AttributionBriefPanel />);
        expect(screen.getByText('ATTRIBUTION BRIEF')).toBeInTheDocument();
    });

    it('should attempt to call API when threat selected', async () => {
        (global.fetch as ReturnType<typeof vi.fn>).mockResolvedValue({
            ok: true,
            json: async () => ({
                summary: 'Test Analysis',
                causes: [],
                suggestedActions: [],
                confidence: 90
            }),
        });

        render(<AttributionBriefPanel />);

        await waitFor(() => {
            expect(global.fetch).toHaveBeenCalledWith('/api/attribution', expect.any(Object));
            expect(mockSetBrief).toHaveBeenCalled();
        });
    });

    it('should fallback to demo mode on API failure', async () => {
        (global.fetch as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('API Fail'));

        render(<AttributionBriefPanel />);

        // Should still eventually call setBrief (with demo data)
        await waitFor(() => {
            expect(mockSetBrief).toHaveBeenCalled();
        });
    });
});
