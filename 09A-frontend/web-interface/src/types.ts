import React from 'react';

export type View = 
  | 'top' 
  | 'phase' 
  | 'decoder' 
  | 'attention' 
  | 'layer' 
  | 'prompts' 
  | 'models';

export interface Model {
  id: string;
  name: string;
  type: string;                                    // e.g. 'Transformer', 'MoE', 'Dense'
  params: string;                                  // e.g. '175B', '32B', '1.2T'
  status: 'Active' | 'Standby' | 'Offline';
  latency: string;                                 // e.g. '42ms'
  energy: string;                                  // e.g. 'Low', 'Medium', 'High', 'Ultra-Low'
}

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
