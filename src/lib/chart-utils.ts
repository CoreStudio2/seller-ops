/**
 * Chart Utilities
 * Mock data generation and formatting helpers for charts
 */

import { format, subHours, subDays } from 'date-fns';

// === MOCK DATA TYPES ===

export interface RevenueDataPoint {
  timestamp: Date;
  revenue: number;
  prediction?: number;
}

export interface RiskDataPoint {
  timestamp: Date;
  riskScore: number;
  severity: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface ThreatDistribution {
  type: string;
  count: number;
  severity: 'CRITICAL' | 'WARNING' | 'INFO' | 'OK';
}

export interface SignalHeatmapData {
  date: Date;
  hour: number;
  intensity: number;
}

// === MOCK DATA GENERATORS ===

/**
 * Generate mock revenue velocity data for last 24 hours
 */
export function generateMockRevenueData(hours: number = 24): RevenueDataPoint[] {
  const data: RevenueDataPoint[] = [];
  const now = new Date();
  const baseRevenue = 2800;
  
  for (let i = hours; i >= 0; i--) {
    const timestamp = subHours(now, i);
    const randomVariation = (Math.random() - 0.5) * 600;
    const trendVariation = Math.sin((i / hours) * Math.PI * 2) * 400;
    const revenue = baseRevenue + randomVariation + trendVariation;
    
    // Add prediction for last 3 hours (future)
    const prediction = i < 3 ? revenue + (Math.random() - 0.3) * 200 : undefined;
    
    data.push({ 
      timestamp, 
      revenue: Math.max(0, revenue),
      prediction: prediction ? Math.max(0, prediction) : undefined
    });
  }
  
  return data;
}

/**
 * Generate mock risk score data for last 7 days
 */
export function generateMockRiskData(days: number = 7): RiskDataPoint[] {
  const data: RiskDataPoint[] = [];
  const now = new Date();
  
  for (let i = days; i >= 0; i--) {
    const timestamp = subDays(now, i);
    const riskScore = Math.floor(Math.random() * 50 + 30); // 30-80
    const severity = riskScore > 70 ? 'HIGH' : riskScore > 50 ? 'MEDIUM' : 'LOW';
    
    data.push({ timestamp, riskScore, severity });
  }
  
  return data;
}

/**
 * Generate mock threat distribution
 */
export function generateMockThreatDistribution(): ThreatDistribution[] {
  const types = [
    { type: 'REFUND_SPIKE', min: 3, max: 15 },
    { type: 'COMPETITOR_PRICE', min: 2, max: 10 },
    { type: 'SEARCH_DROP', min: 5, max: 20 },
    { type: 'CONVERSION_DROP', min: 1, max: 8 },
    { type: 'STOCK_ALERT', min: 4, max: 12 },
  ];
  
  return types.map(({ type, min, max }) => {
    const count = Math.floor(Math.random() * (max - min) + min);
    const severities: Array<'CRITICAL' | 'WARNING' | 'INFO' | 'OK'> = 
      ['CRITICAL', 'WARNING', 'INFO'];
    const severity = severities[Math.floor(Math.random() * severities.length)];
    
    return { type, count, severity };
  });
}

/**
 * Generate signal heatmap data (last 30 days, hourly)
 */
export function generateMockSignalHeatmap(days: number = 30): SignalHeatmapData[] {
  const data: SignalHeatmapData[] = [];
  const now = new Date();
  
  for (let d = days; d >= 0; d--) {
    for (let h = 0; h < 24; h++) {
      const date = subDays(now, d);
      const intensity = Math.floor(Math.random() * 100);
      data.push({ date, hour: h, intensity });
    }
  }
  
  return data;
}

/**
 * Generate mock opportunity trend data
 */
export function generateMockOpportunityData(days: number = 7): { timestamp: Date; opportunityScore: number }[] {
  const data: { timestamp: Date; opportunityScore: number }[] = [];
  const now = new Date();
  
  for (let i = days; i >= 0; i--) {
    const timestamp = subDays(now, i);
    const score = Math.floor(Math.random() * 40 + 40); // 40-80%
    data.push({ timestamp, opportunityScore: score });
  }
  
  return data;
}

/**
 * Generate mock attribution data
 */
export function generateMockAttributionData(): { factor: string; contribution: number; impact: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL' }[] {
  return [
    { factor: 'Price Drop', contribution: -15, impact: 'NEGATIVE' },
    { factor: 'Marketing Campaign', contribution: 25, impact: 'POSITIVE' },
    { factor: 'Competitor Activity', contribution: -10, impact: 'NEGATIVE' },
    { factor: 'Seasonal Demand', contribution: 20, impact: 'POSITIVE' },
    { factor: 'Product Quality', contribution: 15, impact: 'POSITIVE' },
    { factor: 'Shipping Delays', contribution: -8, impact: 'NEGATIVE' },
  ];
}

// === FORMATTERS ===

/**
 * Format currency for Indian Rupees
 */
export function formatCurrency(value: number): string {
  return `₹${value.toFixed(0)}`;
}

/**
 * Format revenue velocity (per hour)
 */
export function formatRevenueVelocity(value: number): string {
  return `${formatCurrency(value)}/hr`;
}

/**
 * Format time for chart tooltips
 */
export function formatChartTime(date: Date): string {
  return format(date, 'MMM dd, HH:mm');
}

/**
 * Format date for axis labels
 */
export function formatAxisDate(date: Date): string {
  return format(date, 'MMM dd');
}

/**
 * Format time for axis labels (hourly)
 */
export function formatAxisTime(date: Date): string {
  return format(date, 'HH:mm');
}

// === COLOR HELPERS ===

/**
 * Get color based on severity
 */
export function getSeverityColor(severity: 'CRITICAL' | 'WARNING' | 'INFO' | 'OK' | 'LOW' | 'MEDIUM' | 'HIGH'): string {
  const colors = {
    CRITICAL: '#ef4444',  // Red
    HIGH: '#ef4444',
    WARNING: '#f59e0b',   // Amber
    MEDIUM: '#f59e0b',
    INFO: '#06b6d4',      // Cyan
    LOW: '#06b6d4',
    OK: '#10b981',        // Green
  };
  return colors[severity] || '#6b7280'; // Gray fallback
}

/**
 * Get risk score color (gradient)
 */
export function getRiskScoreColor(score: number): string {
  if (score >= 70) return '#ef4444'; // Red
  if (score >= 50) return '#f59e0b'; // Amber
  if (score >= 30) return '#06b6d4'; // Cyan
  return '#10b981'; // Green
}
