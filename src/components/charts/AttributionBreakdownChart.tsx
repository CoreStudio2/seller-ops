'use client';

/**
 * Attribution Breakdown Pie Chart
 * Shows revenue change attribution by factor
 */

import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { getSeverityColor } from '@/lib/chart-utils';

interface AttributionFactor {
  factor: string;
  contribution: number; // Percentage
  impact: 'POSITIVE' | 'NEGATIVE' | 'NEUTRAL';
}

interface AttributionBreakdownChartProps {
  data: AttributionFactor[];
  height?: number;
}

const COLORS = {
  POSITIVE: '#10b981',  // Green
  NEGATIVE: '#ef4444',  // Red
  NEUTRAL: '#06b6d4',   // Cyan
};

export function AttributionBreakdownChart({ data, height = 300 }: AttributionBreakdownChartProps) {
  // Transform data for Recharts
  const chartData = data.map(item => ({
    name: item.factor,
    value: Math.abs(item.contribution),
    impact: item.impact,
    color: COLORS[item.impact],
    displayValue: item.contribution, // Keep original sign for tooltip
  }));

  const renderCustomLabel = (entry: any) => {
    return `${entry.value.toFixed(1)}%`;
  };

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie
            data={chartData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={renderCustomLabel}
            outerRadius={100}
            fill="#8884d8"
            dataKey="value"
          >
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#1f2937', 
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#f3f4f6'
            }}
            formatter={(value?: number, name?: string, props?: any) => {
              if (!value || !props) return ['', ''];
              const displayValue = props.payload.displayValue;
              const sign = displayValue >= 0 ? '+' : '';
              return [`${sign}${displayValue.toFixed(1)}%`, props.payload.impact];
            }}
          />
          
          <Legend 
            verticalAlign="bottom"
            height={36}
            iconType="circle"
            wrapperStyle={{ color: '#9ca3af', fontSize: '12px' }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
