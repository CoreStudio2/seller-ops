'use client';

import { useState } from 'react';
import dynamic from 'next/dynamic';
import { LiveStatusBar } from '@/components/metrics/LiveStatusBar';
import { ThreatFeed } from '@/components/feed/ThreatFeed';
import { AttributionBriefPanel } from '@/components/feed/AttributionBrief';
import { BeastModePanel } from '@/components/simulation/BeastModePanel';
import { SmartRecommendationsPanel } from '@/components/recommendations/SmartRecommendationsPanel';
import { useRealtimeDashboard } from '@/lib/hooks';
import { 
  generateMockRevenueData, 
  generateMockThreatDistribution,
  generateMockSignalHeatmap,
  generateMockOpportunityData,
  generateMockAttributionData
} from '@/lib/chart-utils';

// Dynamic import charts for SSR compatibility
const RevenueVelocityChart = dynamic(
  () => import('@/components/charts').then(mod => mod.RevenueVelocityChart),
  { ssr: false }
);
const RiskScoreGauge = dynamic(
  () => import('@/components/charts').then(mod => mod.RiskScoreGauge),
  { ssr: false }
);
const ThreatDistributionChart = dynamic(
  () => import('@/components/charts').then(mod => mod.ThreatDistributionChart),
  { ssr: false }
);
const SignalHeatmap = dynamic(
  () => import('@/components/charts').then(mod => mod.SignalHeatmap),
  { ssr: false }
);
const OpportunityTrendChart = dynamic(
  () => import('@/components/charts').then(mod => mod.OpportunityTrendChart),
  { ssr: false }
);
const AttributionBreakdownChart = dynamic(
  () => import('@/components/charts').then(mod => mod.AttributionBreakdownChart),
  { ssr: false }
);

type ViewMode = 'attribution' | 'beast' | 'recommendations' | 'analytics';

export default function WarRoom() {
  const [viewMode, setViewMode] = useState<ViewMode>('recommendations');
  const [showThreatFeed, setShowThreatFeed] = useState(false);

  // Enable real-time polling
  useRealtimeDashboard();

  // Generate mock data for charts
  const revenueData = generateMockRevenueData(24);
  const threatDistribution = generateMockThreatDistribution();
  const heatmapData = generateMockSignalHeatmap(30);
  const opportunityData = generateMockOpportunityData(7);
  const attributionData = generateMockAttributionData();

  return (
    <div className="min-h-screen flex flex-col bg-surface-base">
      {/* Top Status Bar */}
      <LiveStatusBar />

      {/* Main Content */}
      <div className="flex-1 flex flex-col md:flex-row">
        {/* Left: Threat Feed - Hidden on mobile, visible on tablet+ */}
        <div className={`
          ${showThreatFeed ? 'block' : 'hidden'}
          md:block
          fixed md:relative
          inset-0 md:inset-auto
          z-50 md:z-auto
          bg-surface-base md:bg-transparent
        `}>
          <ThreatFeed onClose={() => setShowThreatFeed(false)} />
        </div>

        {/* Mobile hamburger menu */}
        <button
          onClick={() => setShowThreatFeed(!showThreatFeed)}
          className="md:hidden fixed bottom-4 right-4 z-40 w-14 h-14 bg-signal-cyan text-black rounded-full shadow-lg flex items-center justify-center text-2xl"
          aria-label="Toggle threat feed"
        >
          {showThreatFeed ? '✕' : '≡'}
        </button>

        {/* Center: Main Panel */}
        <main className="flex-1 flex flex-col">
          {/* View Toggle - Scrollable on mobile */}
          <div className="flex border-b border-border-default overflow-x-auto">
            <button
              onClick={() => setViewMode('recommendations')}
              className={`
                flex-1 min-w-[120px] py-3 px-4 font-mono text-xs md:text-sm uppercase tracking-wider transition-colors
                ${viewMode === 'recommendations'
                  ? 'bg-surface-elevated text-signal-cyan border-b-2 border-signal-cyan'
                  : 'text-text-secondary hover:text-text-primary'
                }
              `}
            >
              <span className="mr-2 hidden md:inline">🤖</span>
              <span className="hidden md:inline">Smart Recommendations</span>
              <span className="md:hidden">Recs</span>
            </button>
            <button
              onClick={() => setViewMode('attribution')}
              className={`
                flex-1 min-w-[120px] py-3 px-4 font-mono text-xs md:text-sm uppercase tracking-wider transition-colors
                ${viewMode === 'attribution'
                  ? 'bg-surface-elevated text-signal-green border-b-2 border-signal-green'
                  : 'text-text-secondary hover:text-text-primary'
                }
              `}
            >
              <span className="mr-2 hidden md:inline">◎</span>
              <span className="hidden md:inline">Attribution Brief</span>
              <span className="md:hidden">Attribution</span>
            </button>
            <button
              onClick={() => setViewMode('beast')}
              className={`
                flex-1 min-w-[120px] py-3 px-4 font-mono text-xs md:text-sm uppercase tracking-wider transition-colors
                ${viewMode === 'beast'
                  ? 'bg-surface-elevated text-signal-amber border-b-2 border-signal-amber'
                  : 'text-text-secondary hover:text-text-primary'
                }
              `}
            >
              <span className="mr-2 hidden md:inline">⚔</span>
              <span className="hidden md:inline">Beast Mode</span>
              <span className="md:hidden">Beast</span>
            </button>
            <button
              onClick={() => setViewMode('analytics')}
              className={`
                flex-1 min-w-[120px] py-3 px-4 font-mono text-xs md:text-sm uppercase tracking-wider transition-colors
                ${viewMode === 'analytics'
                  ? 'bg-surface-elevated text-signal-cyan border-b-2 border-signal-cyan'
                  : 'text-text-secondary hover:text-text-primary'
                }
              `}
            >
              <span className="mr-2 hidden md:inline">📊</span>
              <span className="hidden md:inline">Analytics</span>
              <span className="md:hidden">Charts</span>
            </button>
          </div>

          {/* Dynamic Content */}
          {viewMode === 'recommendations' ? (
            <SmartRecommendationsPanel />
          ) : viewMode === 'attribution' ? (
            <AttributionBriefPanel />
          ) : viewMode === 'beast' ? (
            <BeastModePanel />
          ) : (
            <div className="flex-1 p-4 md:p-6 overflow-y-auto">
              <h2 className="text-2xl font-bold text-signal-cyan mb-6 font-mono">
                📊 ANALYTICS DASHBOARD
              </h2>
              
              {/* Charts Grid - Responsive */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Revenue Velocity Chart */}
                <div className="bg-surface-elevated border border-border-default rounded-lg p-4">
                  <h3 className="text-sm font-mono text-text-secondary mb-4 uppercase tracking-wider">
                    Revenue Velocity (24h)
                  </h3>
                  <RevenueVelocityChart data={revenueData} height={250} />
                </div>

                {/* Risk Score Gauge */}
                <div className="bg-surface-elevated border border-border-default rounded-lg p-4">
                  <h3 className="text-sm font-mono text-text-secondary mb-4 uppercase tracking-wider">
                    Current Risk Score
                  </h3>
                  <RiskScoreGauge riskScore={65} height={250} />
                </div>

                {/* Opportunity Trend Chart */}
                <div className="bg-surface-elevated border border-border-default rounded-lg p-4">
                  <h3 className="text-sm font-mono text-text-secondary mb-4 uppercase tracking-wider">
                    Opportunity Score Trend (7 Days)
                  </h3>
                  <OpportunityTrendChart data={opportunityData} height={250} />
                </div>

                {/* Signal Heatmap */}
                <div className="bg-surface-elevated border border-border-default rounded-lg p-4">
                  <h3 className="text-sm font-mono text-text-secondary mb-4 uppercase tracking-wider">
                    Signal Activity Heatmap (30 Days)
                  </h3>
                  <SignalHeatmap data={heatmapData} days={30} height={200} />
                </div>

                {/* Attribution Breakdown Chart */}
                <div className="bg-surface-elevated border border-border-default rounded-lg p-4">
                  <h3 className="text-sm font-mono text-text-secondary mb-4 uppercase tracking-wider">
                    Revenue Attribution Breakdown
                  </h3>
                  <AttributionBreakdownChart data={attributionData} height={280} />
                </div>

                {/* Threat Distribution Chart */}
                <div className="bg-surface-elevated border border-border-default rounded-lg p-4">
                  <h3 className="text-sm font-mono text-text-secondary mb-4 uppercase tracking-wider">
                    Threat Distribution by Type
                  </h3>
                  <ThreatDistributionChart data={threatDistribution} height={280} />
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      {/* Footer Status - Responsive */}
      <footer className="border-t border-border-default px-4 py-2 flex flex-col md:flex-row items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-signal-green animate-pulse" />
          <span className="font-mono text-xs text-text-tertiary">
            Live Feed Connected
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="font-mono text-xs text-text-tertiary hidden md:inline">
            Powered by TensorFlow.js + Gemini AI
          </span>
          <span className="font-mono text-xs text-text-tertiary">
            v0.2.0-beta
          </span>
        </div>
      </footer>
    </div>
  );
}
