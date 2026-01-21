'use client';

/**
 * Revenue Velocity Line Chart
 * Shows hourly revenue trend with optional prediction line
 */

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { formatCurrency, formatAxisTime, type RevenueDataPoint } from '@/lib/chart-utils';

interface RevenueVelocityChartProps {
  data: RevenueDataPoint[];
  height?: number;
  showPrediction?: boolean;
}

export function RevenueVelocityChart({
  data,
  height = 300,
  showPrediction = true
}: RevenueVelocityChartProps) {
  // Transform data for Recharts
  const chartData = data.map(point => ({
    time: formatAxisTime(point.timestamp),
    fullTimestamp: point.timestamp,
    revenue: Math.round(point.revenue),
    prediction: point.prediction ? Math.round(point.prediction) : null,
  }));

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />

          <XAxis
            dataKey="time"
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickLine={{ stroke: '#4b5563' }}
          />

          <YAxis
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
            tickLine={{ stroke: '#4b5563' }}
            tickFormatter={(value) => `₹${(value / 1000).toFixed(1)}k`}
          />

          <Tooltip
            contentStyle={{
              backgroundColor: '#1f2937',
              border: '1px solid #374151',
              borderRadius: '8px',
              color: '#f3f4f6'
            }}
            formatter={(value: number | undefined) => [value != null ? formatCurrency(value) : '', '']}
            labelFormatter={(label, payload) => {
              if (payload && payload[0]) {
                return formatAxisTime(payload[0].payload.fullTimestamp);
              }
              return label;
            }}
          />

          <Legend
            wrapperStyle={{ color: '#9ca3af' }}
            iconType="line"
          />

          {/* Actual revenue line */}
          <Line
            type="monotone"
            dataKey="revenue"
            stroke="#06b6d4"
            strokeWidth={2}
            dot={{ fill: '#06b6d4', r: 3 }}
            activeDot={{ r: 5 }}
            name="Revenue/hr"
          />

          {/* Prediction line (dashed) */}
          {showPrediction && (
            <Line
              type="monotone"
              dataKey="prediction"
              stroke="#f59e0b"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ fill: '#f59e0b', r: 3 }}
              name="Prediction"
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
