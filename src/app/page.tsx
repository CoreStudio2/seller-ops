'use client';

import { useState } from 'react';
import { LiveStatusBar } from '@/components/metrics/LiveStatusBar';
import { ThreatFeed } from '@/components/feed/ThreatFeed';
import { AttributionBriefPanel } from '@/components/feed/AttributionBrief';
import { BeastModePanel } from '@/components/simulation/BeastModePanel';
import { useRealtimeDashboard } from '@/lib/hooks';

type ViewMode = 'attribution' | 'beast';

export default function WarRoom() {
  const [viewMode, setViewMode] = useState<ViewMode>('attribution');

  // Enable real-time polling
  useRealtimeDashboard();

  return (
    <div className="min-h-screen flex flex-col bg-surface-base">
      {/* Top Status Bar */}
      <LiveStatusBar />

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Left: Threat Feed */}
        <ThreatFeed />

        {/* Center: Main Panel */}
        <main className="flex-1 flex flex-col">
          {/* View Toggle */}
          <div className="flex border-b border-border-default">
            <button
              onClick={() => setViewMode('attribution')}
              className={`
                flex-1 py-3 px-4 font-mono text-sm uppercase tracking-wider transition-colors
                ${viewMode === 'attribution'
                  ? 'bg-surface-elevated text-signal-green border-b-2 border-signal-green'
                  : 'text-text-secondary hover:text-text-primary'
                }
              `}
            >
              <span className="mr-2">◎</span>
              Attribution Brief
            </button>
            <button
              onClick={() => setViewMode('beast')}
              className={`
                flex-1 py-3 px-4 font-mono text-sm uppercase tracking-wider transition-colors
                ${viewMode === 'beast'
                  ? 'bg-surface-elevated text-signal-amber border-b-2 border-signal-amber'
                  : 'text-text-secondary hover:text-text-primary'
                }
              `}
            >
              <span className="mr-2">⚔</span>
              Beast Mode
            </button>
          </div>

          {/* Dynamic Content */}
          {viewMode === 'attribution' ? (
            <AttributionBriefPanel />
          ) : (
            <BeastModePanel />
          )}
        </main>
      </div>

      {/* Footer Status */}
      <footer className="border-t border-border-default px-4 py-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-signal-green animate-pulse" />
          <span className="font-mono text-xs text-text-tertiary">
            Live Feed Connected
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="font-mono text-xs text-text-tertiary">
            Powered by Gemini AI
          </span>
          <span className="font-mono text-xs text-text-tertiary">
            v0.1.0-beta
          </span>
        </div>
      </footer>
    </div>
  );
}
