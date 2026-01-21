'use client';

/**
 * Risk Score Radial Gauge
 * Shows current risk score with color-coded severity zones
 */

import { RadialBarChart, RadialBar, Legend, ResponsiveContainer, PolarAngleAxis } from 'recharts';
import { getRiskScoreColor } from '@/lib/chart-utils';

interface RiskScoreGaugeProps {
  riskScore: number; // 0-100
  height?: number;
  showLabel?: boolean;
}

export function RiskScoreGauge({ 
  riskScore, 
  height = 200,
  showLabel = true 
}: RiskScoreGaugeProps) {
  // Clamp score to 0-100
  const score = Math.max(0, Math.min(100, riskScore));
  const color = getRiskScoreColor(score);
  
  // Data for radial bar
  const data = [
    {
      name: 'Risk',
      value: score,
      fill: color,
    },
  ];

  // Get severity text
  const getSeverityText = (score: number) => {
    if (score >= 70) return 'HIGH RISK';
    if (score >= 50) return 'MEDIUM RISK';
    if (score >= 30) return 'ELEVATED';
    return 'LOW RISK';
  };

  return (
    <div className="relative w-full" style={{ height }}>
      <ResponsiveContainer width="100%" height="100%">
        <RadialBarChart
          cx="50%"
          cy="50%"
          innerRadius="60%"
          outerRadius="90%"
          barSize={20}
          data={data}
          startAngle={180}
          endAngle={0}
        >
          <PolarAngleAxis
            type="number"
            domain={[0, 100]}
            angleAxisId={0}
            tick={false}
          />
          
          <RadialBar
            background={{ fill: '#1f2937' }}
            dataKey="value"
            cornerRadius={10}
            fill={color}
          />
        </RadialBarChart>
      </ResponsiveContainer>
      
      {/* Center label */}
      {showLabel && (
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <div className="text-4xl font-bold" style={{ color }}>
            {score}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {getSeverityText(score)}
          </div>
        </div>
      )}
    </div>
  );
}
