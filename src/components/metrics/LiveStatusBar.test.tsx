import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LiveStatusBar } from '@/components/metrics/LiveStatusBar';

// Mock the store
vi.mock('@/lib/store', () => ({
    useLiveStatus: vi.fn(() => null),
}));

describe('LiveStatusBar', () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it('should render the component', () => {
        render(<LiveStatusBar />);

        expect(screen.getByText('SELLEROPS')).toBeInTheDocument();
        expect(screen.getByText('War Room Active')).toBeInTheDocument();
    });

    it('should display revenue velocity metric', () => {
        render(<LiveStatusBar />);

        expect(screen.getByText('Revenue Velocity')).toBeInTheDocument();
    });

    it('should display risk score metric', () => {
        render(<LiveStatusBar />);

        expect(screen.getByText('Risk Score')).toBeInTheDocument();
    });

    it('should display active threats indicator', () => {
        render(<LiveStatusBar />);

        expect(screen.getByText('Active Threats')).toBeInTheDocument();
    });

    it('should display last update timestamp', () => {
        render(<LiveStatusBar />);

        expect(screen.getByText('Last Update')).toBeInTheDocument();
    });

    it('should show SO logo', () => {
        render(<LiveStatusBar />);

        expect(screen.getByText('SO')).toBeInTheDocument();
    });
});
