import { create } from 'zustand';
import type {
    SellerOpsState,
    LiveStatus,
    ThreatEvent,
    AttributionBrief,
    SimulationParams,
    SimulationResult
} from '@/lib/types';

export const useSellerOpsStore = create<SellerOpsState>((set, get) => ({
    // Initial State
    liveStatus: null,
    threats: [],
    selectedThreat: null,
    currentBrief: null,
    isLoadingBrief: false,
    simulationResult: null,
    isSimulating: false,

    // Actions
    setLiveStatus: (status: LiveStatus) => set({ liveStatus: status }),

    addThreat: (threat: ThreatEvent) =>
        set((state) => ({
            threats: [threat, ...state.threats].slice(0, 50) // Keep last 50
        })),

    setThreats: (threats: ThreatEvent[]) => set({ threats }),

    selectThreat: (threat: ThreatEvent | null) =>
        set({ selectedThreat: threat, currentBrief: null }),

    setAttributionBrief: (brief: AttributionBrief | null) =>
        set({ currentBrief: brief, isLoadingBrief: false }),

    runSimulation: async (params: SimulationParams) => {
        set({ isSimulating: true });

        try {
            // Call simulation API
            const response = await fetch('/api/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            });

            if (!response.ok) throw new Error('Simulation failed');

            const result: SimulationResult = await response.json();
            set({ simulationResult: result, isSimulating: false });
        } catch (error) {
            console.error('Simulation error:', error);
            set({ isSimulating: false });
        }
    },
}));

// Selectors for performance optimization
export const useLiveStatus = () => useSellerOpsStore((s) => s.liveStatus);
export const useThreats = () => useSellerOpsStore((s) => s.threats);
export const useSelectedThreat = () => useSellerOpsStore((s) => s.selectedThreat);
export const useAttributionBrief = () => useSellerOpsStore((s) => s.currentBrief);
export const useSimulationResult = () => useSellerOpsStore((s) => s.simulationResult);
