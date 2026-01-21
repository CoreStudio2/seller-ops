'use client';

/**
 * Opportunity Trend Area Chart
 * Shows opportunity score trend over time
 */

import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { formatAxisDate } from '@/lib/chart-utils';

interface OpportunityDataPoint {
  timestamp: Date;
  opportunityScore: number;
}

interface OpportunityTrendChartProps {
  data: OpportunityDataPoint[];
  height?: number;
}

export function OpportunityTrendChart({ data, height = 300 }: OpportunityTrendChartProps) {
  // Transform data for Recharts
  const chartData = data.map(point => ({
    date: formatAxisDate(point.timestamp),
    fullTimestamp: point.timestamp,
    score: point.opportunityScore,
  }));

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <defs>
            <linearGradient id="opportunityGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0.1} />
            </linearGradient>
          </defs>

          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />

          <XAxis
            dataKey="date"
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickLine={{ stroke: '#4b5563' }}
          />

          <YAxis
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickLine={{ stroke: '#4b5563' }}
            domain={[0, 100]}
            tickFormatter={(value) => `${value}%`}
          />

          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#f3f4f6'
            }}
            formatter={(value: number | undefined) => [`${value ?? 0}%`, 'Opportunity']}
            labelFormatter={(label, payload) => {
              if (payload && payload[0]) {
                return formatAxisDate(payload[0].payload.fullTimestamp);
              }
              return label;
            }}
          />

          <Area
            type="monotone"
            dataKey="score"
            stroke="#10b981"
            strokeWidth={2}
            fill="url(#opportunityGradient)"
            dot={{ fill: '#10b981', r: 3 }}
            activeDot={{ r: 5 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
