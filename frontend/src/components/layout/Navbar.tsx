import React, { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Activity, User as UserIcon, LogOut, LogIn, History, Info, Mic } from 'lucide-react';
import { useAuthStore } from '../../store/authStore';
import { api } from '../../api/endpoints';

export const Navbar: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, isAuthenticated, logout } = useAuthStore();
  const [systemHealthy, setSystemHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    api.system.health()
      .then((h) => {
        if (mounted) setSystemHealthy(h.status === 'healthy');
      })
      .catch(() => {
        if (mounted) setSystemHealthy(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const navLinks = [
    { label: 'Analyse', path: '/analyze', icon: Mic },
    { label: 'History', path: '/history', icon: History, requiresAuth: true },
    { label: 'About & Methodology', path: '/about', icon: Info },
  ];

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border-subtle bg-bg-base/85 backdrop-blur-md">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between">
        {/* Brand */}
        <Link to="/" className="flex items-center gap-3.5 group focus:outline-none focus-visible:ring-2 focus-visible:ring-accent rounded-lg p-1">
          <div className="w-12 h-12 rounded-xl bg-bg-surface border border-border-default flex items-center justify-center p-1.5 group-hover:border-accent/40 shadow-sm transition-all">
            <img src="/logo.png" alt="VoiceGuard Logo" className="w-full h-full object-contain" />
          </div>
          <div>
            <span className="font-semibold text-text-primary text-base sm:text-lg tracking-tight font-mono block">VoiceGuard</span>
            <p className="text-xs text-text-tertiary hidden sm:block">Multi-signal audio intelligence</p>
          </div>
        </Link>

        {/* Navigation Links */}
        <nav className="flex items-center gap-1 sm:gap-2">
          {navLinks.map((item) => {
            if (item.requiresAuth && !isAuthenticated) return null;
            const active = location.pathname === item.path;
            const Icon = item.icon;
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  active
                    ? 'bg-bg-elevated text-text-primary border border-border-default'
                    : 'text-text-secondary hover:text-text-primary hover:bg-bg-surface'
                }`}
              >
                <Icon className="w-4 h-4 text-text-tertiary" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Right side: Health & Auth */}
        <div className="flex items-center gap-3">
          {/* Health indicator */}
          <Link
            to="/about"
            title="System Status"
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono bg-bg-surface border border-border-subtle hover:border-border-default transition-colors text-text-tertiary"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                systemHealthy === true
                  ? 'bg-risk-low'
                  : systemHealthy === false
                  ? 'bg-risk-high animate-pulse'
                  : 'bg-risk-unknown'
              }`}
            />
            <span className="hidden md:inline">
              {systemHealthy === true ? 'Pipeline Active' : systemHealthy === false ? 'Degraded' : 'Checking…'}
            </span>
          </Link>

          {/* Auth buttons */}
          {isAuthenticated ? (
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-text-secondary hidden md:inline px-2 py-1 rounded bg-bg-surface border border-border-subtle">
                {user?.display_name || user?.email || 'User'}
              </span>
              <button
                onClick={handleLogout}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium text-text-secondary hover:text-risk-high hover:bg-risk-high-bg transition-colors border border-border-subtle"
                title="Log out"
              >
                <LogOut className="w-4 h-4" />
                <span className="hidden sm:inline">Log out</span>
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <Link
                to="/login"
                className="px-3 py-1.5 rounded-md text-sm font-medium text-text-secondary hover:text-text-primary hover:bg-bg-surface transition-colors"
              >
                Log in
              </Link>
              <Link
                to="/register"
                className="px-3 py-1.5 rounded-md text-sm font-medium bg-accent text-text-inverse hover:bg-accent/90 transition-colors font-semibold"
              >
                Register
              </Link>
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
