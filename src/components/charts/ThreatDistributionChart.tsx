'use client';

/**
 * Threat Distribution Bar Chart
 * Shows count of threats by type, color-coded by severity
 */

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { getSeverityColor, type ThreatDistribution } from '@/lib/chart-utils';

interface ThreatDistributionChartProps {
  data: ThreatDistribution[];
  height?: number;
}

export function ThreatDistributionChart({
  data,
  height = 300
}: ThreatDistributionChartProps) {
  // Transform data for Recharts
  const chartData = data.map(item => ({
    type: item.type.replace(/_/g, ' '),
    count: item.count,
    severity: item.severity,
    color: getSeverityColor(item.severity),
  }));

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart
          data={chartData}
          margin={{ top: 5, right: 30, left: 20, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />

          <XAxis
            dataKey="type"
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            tickLine={{ stroke: '#4b5563' }}
            angle={-45}
            textAnchor="end"
            height={80}
          />

          <YAxis
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickLine={{ stroke: '#4b5563' }}
            label={{
              value: 'Threat Count',
              angle: -90,
              position: 'insideLeft',
              style: { fill: '#9ca3af', fontSize: 12 }
            }}
          />

          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#f3f4f6'
            }}
            formatter={(value: number | undefined, name: string | undefined, props: any) => [
              `${value ?? 0} threats`,
              props.payload.severity
            ]}
            labelStyle={{ color: '#06b6d4' }}
          />

          <Bar
            dataKey="count"
            radius={[8, 8, 0, 0]}
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
