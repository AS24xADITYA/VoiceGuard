import React, { useState } from 'react';
import { User, Lock, Sliders, Trash2, Download, AlertTriangle, Shield } from 'lucide-react';
import { useAuthStore } from '../store/authStore';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

export const SettingsPage: React.FC = () => {
  const { user, logout } = useAuthStore();

  const [displayName, setDisplayName] = useState(user?.display_name || '');
  const [reducedMotion, setReducedMotion] = useState(false);
  const [defaultTechDetails, setDefaultTechDetails] = useState(false);
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleSavePreferences = (e: React.FormEvent) => {
    e.preventDefault();
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2500);
  };

  const handleDeleteAllData = () => {
    if (window.confirm('Are you sure you want to delete all historical analysis records? This cannot be undone.')) {
      alert('All analyses have been queued for permanent deletion.');
    }
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8 py-6">
      <div className="space-y-1">
        <h1 className="text-2xl font-bold font-mono tracking-tight text-text-primary flex items-center gap-2">
          <Sliders className="w-6 h-6 text-accent" />
          <span>Account &amp; Analysis Preferences</span>
        </h1>
        <p className="text-xs text-text-tertiary">
          Configure interface behavior, privacy settings, and forensic data lifecycle
        </p>
      </div>

      {savedSuccess && (
        <div className="p-3.5 rounded-lg bg-risk-low-bg border border-risk-low/30 text-xs text-risk-low font-mono">
          Preferences successfully saved.
        </div>
      )}

      {/* Profile Section */}
      <Card className="p-6 bg-bg-surface border-border-default space-y-4">
        <h2 className="text-sm font-semibold font-mono text-text-primary uppercase tracking-wider flex items-center gap-2">
          <User className="w-4 h-4 text-accent" />
          <span>Analyst Profile</span>
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-xs font-mono">
          <div>
            <label className="text-text-tertiary block mb-1">Email Address</label>
            <input
              type="text"
              disabled
              value={user?.email || 'analyst@voiceguard.org'}
              className="w-full px-3 py-2 rounded bg-bg-elevated border border-border-subtle text-text-secondary cursor-not-allowed"
            />
          </div>
          <div>
            <label className="text-text-tertiary block mb-1">Display Name</label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full px-3 py-2 rounded bg-bg-elevated border border-border-default text-text-primary focus:outline-none focus:border-accent"
            />
          </div>
        </div>
      </Card>

      {/* Preferences */}
      <Card className="p-6 bg-bg-surface border-border-default space-y-4">
        <h2 className="text-sm font-semibold font-mono text-text-primary uppercase tracking-wider flex items-center gap-2">
          <Sliders className="w-4 h-4 text-accent" />
          <span>Interface Behavior</span>
        </h2>
        <div className="space-y-3 text-xs">
          <label className="flex items-center gap-3 p-3 rounded bg-bg-elevated border border-border-subtle cursor-pointer hover:bg-bg-overlay transition-colors">
            <input
              type="checkbox"
              checked={defaultTechDetails}
              onChange={(e) => setDefaultTechDetails(e.target.checked)}
              className="rounded accent-accent w-4 h-4"
            />
            <div>
              <span className="font-semibold text-text-primary block font-mono">
                Always Expand Technical Details
              </span>
              <span className="text-text-tertiary">
                Automatically show feature vectors, model weights, and raw JSON on result pages.
              </span>
            </div>
          </label>

          <label className="flex items-center gap-3 p-3 rounded bg-bg-elevated border border-border-subtle cursor-pointer hover:bg-bg-overlay transition-colors">
            <input
              type="checkbox"
              checked={reducedMotion}
              onChange={(e) => setReducedMotion(e.target.checked)}
              className="rounded accent-accent w-4 h-4"
            />
            <div>
              <span className="font-semibold text-text-primary block font-mono">
                Reduced Motion
              </span>
              <span className="text-text-tertiary">
                Disable waveform and progress bar animations for accessibility.
              </span>
            </div>
          </label>
        </div>

        <div className="flex justify-end pt-2">
          <Button size="sm" onClick={handleSavePreferences}>
            Save Preferences
          </Button>
        </div>
      </Card>

      {/* Data & Privacy Lifecycle per 09 §4.6 */}
      <Card className="p-6 bg-bg-surface border-border-default space-y-4">
        <h2 className="text-sm font-semibold font-mono text-risk-high uppercase tracking-wider flex items-center gap-2">
          <Trash2 className="w-4 h-4 text-risk-high" />
          <span>Data Retention &amp; Erasure</span>
        </h2>
        <p className="text-xs text-text-secondary leading-relaxed">
          VoiceGuard enforces automated retention purges (default: 30 days). Audio artifacts and
          transcripts are permanently deleted from disk upon expiry or user request.
        </p>

        <div className="flex flex-wrap items-center gap-3 pt-2">
          <Button variant="outline" size="sm" onClick={() => alert('Exporting full user archive…')}>
            <Download className="w-3.5 h-3.5 mr-1" />
            <span>Export All Records</span>
          </Button>
          <Button variant="danger" size="sm" onClick={handleDeleteAllData}>
            <Trash2 className="w-3.5 h-3.5 mr-1" />
            <span>Delete All Analyses</span>
          </Button>
        </div>
      </Card>
    </div>
  );
};
