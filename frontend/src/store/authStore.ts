import { create } from 'zustand';

export interface User {
  id: string;
  email: string;
  display_name: string;
  role: 'user' | 'admin';
}

interface AuthState {
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  setAuth: (user: User, accessToken: string, refreshToken: string) => void;
  setTokens: (accessToken: string, refreshToken: string) => void;
  setUser: (user: User) => void;
  logout: () => void;
}

const STORAGE_KEY_REFRESH = 'voiceguard_refresh_token';
const STORAGE_KEY_USER = 'voiceguard_user';

export const useAuthStore = create<AuthState>((set) => {
  // Hydrate from localStorage on initial load
  const initialRefresh = localStorage.getItem(STORAGE_KEY_REFRESH);
  let initialUser: User | null = null;
  try {
    const rawUser = localStorage.getItem(STORAGE_KEY_USER);
    if (rawUser) initialUser = JSON.parse(rawUser);
  } catch {
    // Ignore JSON parse errors
  }

  return {
    user: initialUser,
    accessToken: null, // Access tokens kept in memory
    refreshToken: initialRefresh,
    isAuthenticated: !!initialRefresh,

    setAuth: (user, accessToken, refreshToken) => {
      localStorage.setItem(STORAGE_KEY_REFRESH, refreshToken);
      localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(user));
      set({
        user,
        accessToken,
        refreshToken,
        isAuthenticated: true,
      });
    },

    setTokens: (accessToken, refreshToken) => {
      localStorage.setItem(STORAGE_KEY_REFRESH, refreshToken);
      set({
        accessToken,
        refreshToken,
        isAuthenticated: true,
      });
    },

    setUser: (user) => {
      localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(user));
      set({ user });
    },

    logout: () => {
      localStorage.removeItem(STORAGE_KEY_REFRESH);
      localStorage.removeItem(STORAGE_KEY_USER);
      set({
        user: null,
        accessToken: null,
        refreshToken: null,
        isAuthenticated: false,
      });
    },
  };
});
