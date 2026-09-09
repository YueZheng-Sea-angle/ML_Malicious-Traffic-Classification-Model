import { Activity, Boxes, ShieldCheck, Upload } from "lucide-react";
import { useEffect, useState } from "react";
import { NavLink, Navigate, Route, Routes } from "react-router-dom";

import { Badge } from "@/components/ui/badge";
import { api, type Health } from "@/lib/api";
import { cn } from "@/lib/utils";
import ModelsPage from "@/pages/ModelsPage";
import ResultsPage from "@/pages/ResultsPage";
import UploadPage from "@/pages/UploadPage";

const NAV = [
  { to: "/upload", label: "流量上传", icon: Upload },
  { to: "/results", label: "结果展示", icon: Activity },
  { to: "/models", label: "模型管理", icon: Boxes },
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e: Error) => setError(e.message));
  }, []);

  return (
    <div className="flex min-h-full flex-col">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center gap-6 px-6 py-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-6 w-6 text-brand-600" />
            <span className="text-lg font-semibold text-slate-900">加密代理工具识别系统</span>
            <span className="text-xs text-slate-400">DataCon T1 · 11 类代理/隧道工具</span>
          </div>
          <nav className="flex items-center gap-1">
            {NAV.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  cn(
                    "flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-brand-50 text-brand-700"
                      : "text-slate-600 hover:bg-slate-100",
                  )
                }
              >
                <Icon className="h-4 w-4" />
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto">
            <ServiceBadge health={health} error={error} />
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-6">
        <Routes>
          <Route path="/" element={<Navigate to="/upload" replace />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="*" element={<p className="text-slate-500">页面不存在</p>} />
        </Routes>
      </main>

      <footer className="border-t border-slate-200 bg-white py-3 text-center text-xs text-slate-400">
        基于深度学习的加密代理/隧道流量工具识别与特征分析
      </footer>
    </div>
  );
}

function ServiceBadge({ health, error }: { health: Health | null; error: string | null }) {
  if (error) return <Badge tone="danger">后端未连接</Badge>;
  if (!health) return <Badge tone="neutral">检测中…</Badge>;
  const mode = health.inference_mode;
  return (
    <div className="flex items-center gap-2">
      <Badge tone="success">后端 v{health.version}</Badge>
      <Badge tone={mode === "model" ? "info" : "neutral"}>
        {mode === "model" ? "已加载模型权重" : "模型未加载"}
      </Badge>
    </div>
  );
}
