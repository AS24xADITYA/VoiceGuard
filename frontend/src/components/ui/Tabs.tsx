import React from 'react';

export interface TabItem {
  id: string;
  label: string;
  icon?: React.ComponentType<{ className?: string }>;
  count?: number;
}

export interface TabsProps {
  tabs: TabItem[];
  activeTab: string;
  onChange: (id: string) => void;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({ tabs, activeTab, onChange, className = '' }) => {
  return (
    <div className={`flex border-b border-border-default space-x-2 ${className}`} role="tablist">
      {tabs.map((tab) => {
        const isActive = activeTab === tab.id;
        const Icon = tab.icon;
        return (
          <button
            key={tab.id}
            role="tab"
            aria-selected={isActive}
            onClick={() => onChange(tab.id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium border-b-2 transition-all duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent -mb-[1px] ${
              isActive
                ? 'border-accent text-accent'
                : 'border-transparent text-text-secondary hover:text-text-primary hover:border-border-strong'
            }`}
          >
            {Icon && <Icon className={`w-4 h-4 ${isActive ? 'text-accent' : 'text-text-tertiary'}`} />}
            <span>{tab.label}</span>
            {tab.count !== undefined && (
              <span
                className={`px-1.5 py-0.5 rounded text-[11px] font-mono ${
                  isActive ? 'bg-accent/20 text-accent' : 'bg-bg-elevated text-text-tertiary'
                }`}
              >
                {tab.count}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
};
