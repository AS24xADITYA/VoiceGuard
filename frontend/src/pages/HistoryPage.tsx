import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
  History,
  Trash2,
  Filter,
  ArrowUpDown,
  Mic,
  FileAudio,
  LogIn,
  AlertCircle,
  ExternalLink,
} from 'lucide-react';
import { useAuthStore } from '../store/authStore';
import { api } from '../api/endpoints';
import { AnalysisSummaryItem, VerdictType } from '../types/api';
import { VerdictBadge } from '../components/ui/Badge';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';

export const HistoryPage: React.FC = () => {
  const navigate = useNavigate();
  const { isAuthenticated } = useAuthStore();

  const [items, setItems] = useState<AnalysisSummaryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [verdictFilter, setVerdictFilter] = useState<string>('ALL');

  const fetchHistory = async () => {
    if (!isAuthenticated) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.analyses.list({
        verdict: verdictFilter === 'ALL' ? undefined : verdictFilter,
        page: 1,
        page_size: 50,
      });
      setItems(res.items);
      setTotal(res.total);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isAuthenticated) {
      fetchHistory();
    } else {
      setLoading(false);
    }
  }, [isAuthenticated, verdictFilter]);

  const handleDelete = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    if (!window.confirm('Delete this analysis record?')) return;
    try {
      await api.analyses.delete(id);
      setItems((prev) => prev.filter((item) => item.id !== id));
      setTotal((prev) => prev - 1);
    } catch (err: any) {
      alert('Failed to delete analysis');
    }
  };

  // Guest users state per 09 §4.5
  if (!isAuthenticated) {
    return (
      <div className="py-16 max-w-lg mx-auto text-center space-y-6">
        <div className="w-14 h-14 rounded-2xl bg-bg-surface border border-border-default flex items-center justify-center text-accent mx-auto">
          <History className="w-7 h-7" />
        </div>
        <div className="space-y-2">
          <h2 className="text-2xl font-bold font-mono tracking-tight text-text-primary">
            Analysis History Requires an Account
          </h2>
          <p className="text-xs text-text-secondary leading-relaxed">
            Guest analyses are processed anonymously in memory and purged according to data retention
            policies. To save, review, and export longitudinal records, log in or create an account.
          </p>
        </div>
        <div className="flex items-center justify-center gap-3 pt-2">
          <Link to="/login">
            <Button variant="primary">Log In</Button>
          </Link>
          <Link to="/register">
            <Button variant="secondary">Create Free Account</Button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 py-4">
      {/* Header & Filter Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border-subtle">
        <div>
          <h1 className="text-2xl font-bold font-mono tracking-tight text-text-primary flex items-center gap-2.5">
            <History className="w-6 h-6 text-accent" />
            <span>Analysis History</span>
          </h1>
          <p className="text-xs text-text-tertiary mt-1">
            Archived evaluations, signal breakdowns, and forensic audit trails ({total} records)
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-2 overflow-x-auto pb-1">
          <Filter className="w-3.5 h-3.5 text-text-tertiary shrink-0" />
          {['ALL', 'HIGH', 'MODERATE', 'LOW', 'INCONCLUSIVE'].map((v) => (
            <button
              key={v}
              onClick={() => setVerdictFilter(v)}
              className={`px-2.5 py-1 rounded text-xs font-mono transition-colors border ${
                verdictFilter === v
                  ? 'bg-accent text-text-inverse border-accent font-semibold'
                  : 'bg-bg-elevated border-border-subtle text-text-secondary hover:bg-bg-overlay'
              }`}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      {loading ? (
        <div className="py-20 text-center space-y-3">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-xs font-mono text-text-tertiary">Retrieving history entries…</p>
        </div>
      ) : error ? (
        <div className="p-4 rounded-lg bg-risk-high-bg border border-risk-high/30 text-risk-high text-xs">
          {error}
        </div>
      ) : items.length === 0 ? (
        <div className="py-20 text-center space-y-4 max-w-md mx-auto">
          <FileAudio className="w-12 h-12 text-text-tertiary mx-auto opacity-40" />
          <div>
            <h3 className="text-base font-semibold text-text-primary font-mono">No Analyses Found</h3>
            <p className="text-xs text-text-tertiary mt-1">
              You have not submitted any audio files under this account yet.
            </p>
          </div>
          <Link to="/analyze">
            <Button variant="primary" size="sm">
              <Mic className="w-4 h-4 mr-1" />
              <span>Submit First Analysis</span>
            </Button>
          </Link>
        </div>
      ) : (
        /* History Table per 09 §4.5 */
        <div className="rounded-xl border border-border-default bg-bg-surface overflow-hidden shadow-card">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-bg-elevated/70 border-b border-border-subtle text-text-tertiary uppercase tracking-wider">
                <tr>
                  <th className="py-3 px-4">Filename / Label</th>
                  <th className="py-3 px-4">Verdict</th>
                  <th className="py-3 px-4">Risk %</th>
                  <th className="py-3 px-4">Language</th>
                  <th className="py-3 px-4">Duration</th>
                  <th className="py-3 px-4">Date</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-subtle/50 text-text-secondary">
                {items.map((item) => (
                  <tr
                    key={item.id}
                    onClick={() => navigate(`/analysis/${item.id}`)}
                    className="hover:bg-bg-elevated/50 cursor-pointer transition-colors group"
                  >
                    <td className="py-3 px-4 text-text-primary font-sans font-medium flex items-center gap-2">
                      <FileAudio className="w-4 h-4 text-accent shrink-0" />
                      <span className="truncate max-w-[200px]" title={item.original_filename || item.id}>
                        {item.original_filename || `Analysis ${item.id.substring(0, 8)}`}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      <VerdictBadge verdict={item.verdict || 'INCONCLUSIVE'} size="sm" />
                    </td>
                    <td className="py-3 px-4 font-bold text-text-primary">
                      {item.risk_probability !== null && item.risk_probability !== undefined
                        ? `${Math.round(item.risk_probability * 100)}%`
                        : '—'}
                    </td>
                    <td className="py-3 px-4 uppercase text-text-tertiary">
                      {item.detected_language || 'EN'}
                    </td>
                    <td className="py-3 px-4 text-text-tertiary">
                      {item.duration_seconds ? `${item.duration_seconds.toFixed(1)}s` : '—'}
                    </td>
                    <td className="py-3 px-4 text-text-tertiary">
                      {new Date(item.created_at).toLocaleDateString()}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={(e) => handleDelete(e, item.id)}
                        className="p-1.5 rounded text-text-tertiary hover:text-risk-high hover:bg-risk-high-bg transition-colors"
                        title="Delete record"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
