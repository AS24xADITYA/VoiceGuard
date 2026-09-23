import React from 'react';
import { Navbar } from './Navbar';
import { Footer } from './Footer';

interface ShellProps {
  children: React.ReactNode;
}

export const Shell: React.FC<ShellProps> = ({ children }) => {
  return (
    <div className="min-h-screen flex flex-col bg-bg-base text-text-primary antialiased">
      <Navbar />
      <main className="flex-1 w-full mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 relative z-10">
        {children}
      </main>
      <Footer />
    </div>
  );
};
