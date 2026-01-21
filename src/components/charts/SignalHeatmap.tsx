'use client';

/**
 * Signal Heatmap Calendar
 * Shows signal intensity over time (GitHub-style contribution graph)
 */

import { useMemo } from 'react';
import { format, startOfWeek, endOfWeek, eachDayOfInterval, subDays } from 'date-fns';
import { type SignalHeatmapData } from '@/lib/chart-utils';

interface SignalHeatmapProps {
  data: SignalHeatmapData[];
  days?: number;
  height?: number;
}

export function SignalHeatmap({ data, days = 30, height = 150 }: SignalHeatmapProps) {
  // Group data by date
  const heatmapData = useMemo(() => {
    const now = new Date();
    const startDate = subDays(now, days);
    const dateRange = eachDayOfInterval({ start: startDate, end: now });

    // Create map of date -> average intensity
    // Create map of date -> list of intensities
    const intensityMap = new Map<string, number[]>();

    data.forEach(point => {
      const dateKey = format(point.date, 'yyyy-MM-dd');
      const current = intensityMap.get(dateKey) || [];
      intensityMap.set(dateKey, [...current, point.intensity]);
    });

    // Calculate average intensity per day
    return dateRange.map(date => {
      const dateKey = format(date, 'yyyy-MM-dd');
      const intensities = intensityMap.get(dateKey) || [];
      const avgIntensity = intensities.length > 0
        ? intensities.reduce((a, b) => a + b, 0) / intensities.length
        : 0;

      return {
        date,
        dateKey,
        intensity: avgIntensity,
      };
    });
  }, [data, days]);

  // Get color based on intensity
  const getColor = (intensity: number) => {
    if (intensity === 0) return '#1f2937'; // Gray-800
    if (intensity < 25) return '#065f46'; // Green-800
    if (intensity < 50) return '#047857'; // Green-700
    if (intensity < 75) return '#059669'; // Green-600
    return '#10b981'; // Green-500
  };

  // Group by weeks
  const weeks = useMemo(() => {
    const result: typeof heatmapData[] = [];
    let currentWeek: typeof heatmapData = [];

    heatmapData.forEach((day, index) => {
      currentWeek.push(day);
      if ((index + 1) % 7 === 0 || index === heatmapData.length - 1) {
        result.push([...currentWeek]);
        currentWeek = [];
      }
    });

    return result;
  }, [heatmapData]);

  const cellSize = 12;
  const cellGap = 3;

  return (
    <div className="w-full overflow-x-auto" style={{ height }}>
      <div className="inline-flex gap-1">
        {weeks.map((week, weekIndex) => (
          <div key={weekIndex} className="flex flex-col gap-1">
            {week.map((day) => (
              <div
                key={day.dateKey}
                className="rounded-sm transition-all hover:ring-2 hover:ring-signal-cyan cursor-pointer"
                style={{
                  width: cellSize,
                  height: cellSize,
                  backgroundColor: getColor(day.intensity),
                }}
                title={`${format(day.date, 'MMM dd, yyyy')}: ${Math.round(day.intensity)} signals`}
              />
            ))}
          </div>
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center gap-2 mt-4 text-xs text-text-tertiary font-mono">
        <span>Less</span>
        {[0, 20, 40, 60, 80].map(intensity => (
          <div
            key={intensity}
            className="rounded-sm"
            style={{
              width: cellSize,
              height: cellSize,
              backgroundColor: getColor(intensity),
            }}
          />
        ))}
        <span>More</span>
      </div>
    </div>
  );
}
