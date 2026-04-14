import React from 'react';

export type View = 
  | 'top' 
  | 'phase' 
  | 'decoder' 
  | 'attention' 
  | 'layer' 
  | 'prompts' 
  | 'models';

export interface MetricCardProps {
  title: string;
  value: string;
  unit?: string;
  trend?: string;
  icon: React.ReactNode;
  color?: string;
  progress?: number;
  subtext?: string;
}
