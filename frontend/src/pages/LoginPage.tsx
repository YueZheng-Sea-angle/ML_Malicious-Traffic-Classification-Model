import { Loader2, Lock, LogIn, ShieldCheck } from "lucide-react";
import { FormEvent, useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const { user, login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const from = (location.state as { from?: string } | null)?.from ?? "/upload";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (user) {
    return <Navigate to={from} replace />;
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      await login(username.trim(), password);
      navigate(from, { replace: true });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-shell">
      <div className="auth-panel mx-auto w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-600 text-white shadow-lg shadow-brand-600/30">
            <ShieldCheck className="h-7 w-7" />
          </div>
          <h1 className="text-2xl font-bold text-slate-900">欢迎回来</h1>
          <p className="mt-2 text-sm text-slate-500">登录 MalFlow 加密代理工具识别系统</p>
        </div>

        <Card className="border-0 shadow-xl shadow-slate-200/60">
          <CardHeader title="账号登录" description="演示账号：demo / demo123456" />
          <CardBody>
            <form className="space-y-4" onSubmit={(e) => void handleSubmit(e)}>
              <Input
                label="用户名"
                name="username"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="请输入用户名"
                required
              />
              <Input
                label="密码"
                name="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="请输入密码"
                required
              />

              {error ? (
                <p className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</p>
              ) : null}

              <Button type="submit" className="w-full" disabled={submitting}>
                {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <LogIn className="h-4 w-4" />}
                登录
              </Button>
            </form>

            <p className="mt-6 text-center text-sm text-slate-500">
              还没有账号？{" "}
              <Link to="/register" className="font-medium text-brand-600 hover:text-brand-700">
                立即注册
              </Link>
            </p>
          </CardBody>
        </Card>

        <p className="mt-6 flex items-center justify-center gap-2 text-xs text-slate-400">
          <Lock className="h-3.5 w-3.5" />
          会话采用 Bearer Token，密码 PBKDF2 加密存储
        </p>
      </div>
    </div>
  );
}
