/**
 * SellerOps Core Types
 * Type definitions for the Seller War Room system
 */

// === BRANDED TYPES (Domain primitives) ===
export type SignalId = string & { __brand: 'SignalId' };
export type ThreatId = string & { __brand: 'ThreatId' };
export type SKUId = string & { __brand: 'SKUId' };

// === SIGNAL TYPES ===
export type SignalType =
    | 'PRICE_CHANGE'
    | 'REFUND_SPIKE'
    | 'CART_ABANDONMENT'
    | 'SHIPPING_DELAY'
    | 'COMPETITOR_PRICE'
    | 'SEARCH_DROP'
    | 'CONVERSION_DROP'
    | 'STOCK_ALERT';

export type SeverityLevel = 'CRITICAL' | 'WARNING' | 'INFO' | 'OK';

export interface Signal {
    id: SignalId;
    type: SignalType;
    timestamp: Date;
    skuId?: SKUId;
    value: number;
    previousValue?: number;
    delta?: number;
    metadata?: Record<string, unknown>;
}

// === THREAT EVENT ===
export interface ThreatEvent {
    id: ThreatId;
    severity: SeverityLevel;
    title: string;
    description: string;
    signals: SignalId[];
    detectedAt: Date;
    suggestedAction?: string;
    confidence: number; // 0-100
}

// === ATTRIBUTION BRIEF (Gemini Output) ===
export interface AttributionBrief {
    threatId: ThreatId;
    summary: string;
    causes: CausalFactor[];
    suggestedActions: SuggestedAction[];
    confidence: number;
    generatedAt: Date;
}

export interface CausalFactor {
    factor: string;
    impact: 'HIGH' | 'MEDIUM' | 'LOW';
    evidence: string;
}

export interface SuggestedAction {
    action: string;
    priority: 'IMMEDIATE' | 'SOON' | 'OPTIONAL';
    expectedOutcome: string;
}

// === SIMULATION (Beast Mode) ===
export interface SimulationParams {
    priceChange: number;      // Percentage change (-100 to +100)
    adSpendChange: number;    // Percentage change
    shippingSpeedChange: number; // Days delta
}

export interface SimulationResult {
    params: SimulationParams;
    projectedRevenue: number;
    projectedMargin: number;
    projectedVolume: number;
    competitorResponseProbability: number; // 0-100
    riskLevel: SeverityLevel;
    timestamp: Date;
}

// === LIVE STATUS ===
export interface LiveStatus {
    revenueVelocity: number;      // Revenue per hour
    revenueDirection: 'UP' | 'DOWN' | 'STABLE';
    riskScore: number;            // 0-100
    opportunityScore: number;     // 0-100
    activeThreats: number;
    lastUpdated: Date;
}

// === STORE STATE (Zustand) ===
export interface SellerOpsState {
    // Live Status
    liveStatus: LiveStatus | null;

    // Threat Feed
    threats: ThreatEvent[];
    selectedThreat: ThreatEvent | null;

    // Attribution
    currentBrief: AttributionBrief | null;
    isLoadingBrief: boolean;

    // Simulation
    simulationResult: SimulationResult | null;
    isSimulating: boolean;

    // Actions
    setLiveStatus: (status: LiveStatus) => void;
    setThreats: (threat: ThreatEvent[]) => void;
    addThreat: (threat: ThreatEvent) => void;
    selectThreat: (threat: ThreatEvent | null) => void;
    setAttributionBrief: (brief: AttributionBrief | null) => void;
    runSimulation: (params: SimulationParams) => Promise<void>;
}
