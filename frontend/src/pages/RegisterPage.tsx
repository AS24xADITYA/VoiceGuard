import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Shield, Lock, Mail, User, AlertTriangle, ArrowRight } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { useAuthStore } from '../store/authStore';
import { api } from '../api/endpoints';

export const RegisterPage: React.FC = () => {
  const navigate = useNavigate();
  const { setAuth } = useAuthStore();

  const [email, setEmail] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!email || !password) {
      setError('Please fill in email and password.');
      return;
    }
    if (password.length < 8) {
      setError('Password must be at least 8 characters long.');
      return;
    }
    if (password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await api.auth.register({
        email,
        password,
        display_name: displayName || undefined,
      });

      // Automatically log in after registration
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);
      const loginRes = await api.auth.login(formData);

      setAuth(loginRes.user, loginRes.access_token, loginRes.refresh_token);
      navigate('/history');
    } catch (err: any) {
      setError(
        err.response?.data?.detail || err.message || 'Registration failed. Email might already be in use.'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="py-12 max-w-md mx-auto space-y-6">
      <div className="text-center space-y-2">
        <div className="w-12 h-12 rounded-xl bg-bg-surface border border-border-default flex items-center justify-center text-accent mx-auto">
          <Shield className="w-6 h-6" />
        </div>
        <h1 className="text-2xl font-bold font-mono tracking-tight text-text-primary">
          Create an Account
        </h1>
        <p className="text-xs text-text-tertiary">
          Join VoiceGuard to preserve forensic records and run challenge evaluations
        </p>
      </div>

      <Card className="p-6 bg-bg-surface border-border-default shadow-card">
        <form onSubmit={handleSubmit} className="space-y-4">
          {error && (
            <div className="p-3 rounded-md bg-risk-high-bg border border-risk-high/30 text-xs text-risk-high flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <div className="space-y-1.5">
            <label className="text-xs font-mono text-text-secondary block" htmlFor="displayName">
              Display Name (optional)
            </label>
            <div className="relative">
              <User className="w-4 h-4 text-text-tertiary absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                id="displayName"
                type="text"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Security Analyst"
                className="w-full pl-9 pr-3 py-2 rounded-md bg-bg-elevated border border-border-default text-text-primary text-sm focus:outline-none focus:border-accent font-mono transition-colors placeholder:text-text-tertiary"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-mono text-text-secondary block" htmlFor="email">
              Email Address
            </label>
            <div className="relative">
              <Mail className="w-4 h-4 text-text-tertiary absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="analyst@voiceguard.org"
                className="w-full pl-9 pr-3 py-2 rounded-md bg-bg-elevated border border-border-default text-text-primary text-sm focus:outline-none focus:border-accent font-mono transition-colors placeholder:text-text-tertiary"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-mono text-text-secondary block" htmlFor="password">
              Password (min 8 characters)
            </label>
            <div className="relative">
              <Lock className="w-4 h-4 text-text-tertiary absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                id="password"
                type="password"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full pl-9 pr-3 py-2 rounded-md bg-bg-elevated border border-border-default text-text-primary text-sm focus:outline-none focus:border-accent font-mono transition-colors"
              />
            </div>
          </div>

          <div className="space-y-1.5">
            <label className="text-xs font-mono text-text-secondary block" htmlFor="confirmPassword">
              Confirm Password
            </label>
            <div className="relative">
              <Lock className="w-4 h-4 text-text-tertiary absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
              <input
                id="confirmPassword"
                type="password"
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="••••••••••••"
                className="w-full pl-9 pr-3 py-2 rounded-md bg-bg-elevated border border-border-default text-text-primary text-sm focus:outline-none focus:border-accent font-mono transition-colors"
              />
            </div>
          </div>

          <Button type="submit" loading={loading} className="w-full mt-2">
            <span>Register &amp; Continue</span>
            <ArrowRight className="w-4 h-4 ml-1" />
          </Button>
        </form>

        <div className="mt-6 pt-4 border-t border-border-subtle text-center text-xs text-text-tertiary">
          <span>Already have an account? </span>
          <Link to="/login" className="text-accent hover:underline font-medium">
            Log in instead
          </Link>
        </div>
      </Card>
    </div>
  );
};
