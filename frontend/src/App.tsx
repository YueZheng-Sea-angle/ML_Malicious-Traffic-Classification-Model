import { Activity, Boxes, FlaskConical, LogOut, ShieldCheck, Upload, UserCircle2 } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Navigate, Outlet, Route, Routes } from "react-router-dom";

import ProtectedRoute from "@/components/ProtectedRoute";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/lib/auth";
import { api, type Health } from "@/lib/api";
import { cn } from "@/lib/utils";
import EvaluatePage from "@/pages/EvaluatePage";
import LoginPage from "@/pages/LoginPage";
import ModelsPage from "@/pages/ModelsPage";
import ProfilePage from "@/pages/ProfilePage";
import RegisterPage from "@/pages/RegisterPage";
import ResultsPage from "@/pages/ResultsPage";
import UploadPage from "@/pages/UploadPage";

const NAV = [
  { to: "/upload", label: "流量上传", icon: Upload },
  { to: "/results", label: "结果展示", icon: Activity },
  { to: "/models", label: "模型管理", icon: Boxes },
  { to: "/evaluate", label: "模型测试", icon: FlaskConical },
];

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/register" element={<RegisterPage />} />

      <Route element={<ProtectedRoute />}>
        <Route element={<AppLayout />}>
          <Route path="/" element={<Navigate to="/upload" replace />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/evaluate" element={<EvaluatePage />} />
          <Route path="/profile" element={<ProfilePage />} />
          <Route path="*" element={<p className="text-slate-500">页面不存在</p>} />
        </Route>
      </Route>
    </Routes>
  );
}

function AppLayout() {
  const { user, logout } = useAuth();
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="app-shell flex min-h-full flex-col">
      <header className="sticky top-0 z-20 border-b border-slate-200/80 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-4 px-6 py-3">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow-md shadow-brand-600/20">
              <ShieldCheck className="h-5 w-5" />
            </div>
            <div>
              <p className="text-base font-semibold text-slate-900">加密代理工具识别系统</p>
              <p className="text-xs text-slate-400">DataCon T1 · 11 类代理/隧道工具</p>
            </div>
          </div>

          <nav className="flex flex-1 flex-wrap items-center gap-1">
            {NAV.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-brand-50 text-brand-700 shadow-sm"
                      : "text-slate-600 hover:bg-slate-100",
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {label}
              </NavLink>
            ))}
          </nav>

          <div className="flex items-center gap-3">
            <ServiceBadge health={health} error={error} />
            {user ? (
              <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-2 py-1">
                <NavLink
                  to="/profile"
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm transition-colors",
                      isActive ? "bg-white text-brand-700 shadow-sm" : "text-slate-700 hover:bg-white",
                    )
                  }
                >
                  <UserCircle2 className="h-4 w-4" />
                  <span className="max-w-[120px] truncate">{user.display_name}</span>
                </NavLink>
                <Button variant="ghost" size="sm" onClick={() => void logout()} title="退出登录">
                  <LogOut className="h-4 w-4" />
                </Button>
              </div>
            ) : null}
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-6">
        <Outlet />
      </main>

      <footer className="border-t border-slate-200 bg-white/80 py-4 text-center text-xs text-slate-400">
        基于深度学习的加密代理/隧道流量工具识别与特征分析 · MalFlow v0.1
      </footer>
    </div>
  );
}

function ServiceBadge({ health, error }: { health: Health | null; error: string | null }) {
  if (error) return <Badge tone="danger">后端未连接</Badge>;
  if (!health) return <Badge tone="neutral">检测中…</Badge>;
  const mode = health.inference_mode;
  return (
    <div className="hidden items-center gap-2 sm:flex">
      <Badge tone="success">后端 v{health.version}</Badge>
      <Badge tone={mode === "model" ? "info" : "neutral"}>
        {mode === "model" ? "已加载模型" : "模型未加载"}
      </Badge>
    </div>
  );
}
