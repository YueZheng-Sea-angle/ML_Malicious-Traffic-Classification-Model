import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { api, clearAuthToken, getAuthToken, setAuthToken, type User } from "@/lib/api";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (payload: {
    username: string;
    email: string;
    password: string;
    display_name?: string;
  }) => Promise<void>;
  logout: () => Promise<void>;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refreshUser = useCallback(async () => {
    const token = getAuthToken();
    if (!token) {
      setUser(null);
      return;
    }
    try {
      const profile = await api.me();
      setUser(profile);
    } catch {
      clearAuthToken();
      setUser(null);
    }
  }, []);

  useEffect(() => {
    void (async () => {
      await refreshUser();
      setLoading(false);
    })();
  }, [refreshUser]);

  const login = useCallback(async (username: string, password: string) => {
    const response = await api.login({ username, password });
    setAuthToken(response.access_token);
    setUser(response.user);
  }, []);

  const register = useCallback(
    async (payload: {
      username: string;
      email: string;
      password: string;
      display_name?: string;
    }) => {
      const response = await api.register(payload);
      setAuthToken(response.access_token);
      setUser(response.user);
    },
    [],
  );

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      // 忽略网络错误，本地仍清除会话
    } finally {
      clearAuthToken();
      setUser(null);
    }
  }, []);

  const value = useMemo(
    () => ({ user, loading, login, register, logout, refreshUser }),
    [user, loading, login, register, logout, refreshUser],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth 必须在 AuthProvider 内使用");
  }
  return context;
}
