import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ThreatFeed } from '@/components/feed/ThreatFeed';

// Mock the store
const mockSelectThreat = vi.fn();

vi.mock('@/lib/store', () => ({
    useThreats: vi.fn(() => []),
    useSelectedThreat: vi.fn(() => null),
    useSellerOpsStore: vi.fn((selector) => {
        if (typeof selector === 'function') {
            return selector({ selectThreat: mockSelectThreat });
        }
        return { selectThreat: mockSelectThreat };
    }),
}));

describe('ThreatFeed', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('should render the feed header', () => {
        render(<ThreatFeed />);

        expect(screen.getByText('Threat Feed')).toBeInTheDocument();
    });

    it('should show demo threats when no data provided', () => {
        render(<ThreatFeed />);

        // Should show demo threat cards
        expect(screen.getByText('Buy Box Lost - SKU #334')).toBeInTheDocument();
        expect(screen.getByText('Refund Spike Detected')).toBeInTheDocument();
    });

    it('should display severity badges', () => {
        render(<ThreatFeed />);

        // Should have CRITICAL and WARNING badges
        expect(screen.getAllByText('CRITICAL').length).toBeGreaterThan(0);
        expect(screen.getAllByText('WARNING').length).toBeGreaterThan(0);
    });

    it('should show View All History button', () => {
        render(<ThreatFeed />);

        expect(screen.getByText('View All History →')).toBeInTheDocument();
    });

    it('should display confidence levels', () => {
        render(<ThreatFeed />);

        // Demo threats have confidence of 94, 87, etc.
        expect(screen.getByText(/Confidence: 94%/)).toBeInTheDocument();
    });

    it('should show suggested actions when available', () => {
        render(<ThreatFeed />);

        expect(screen.getByText(/Match price to regain Buy Box/)).toBeInTheDocument();
    });
});
