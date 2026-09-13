import { Loader2, ShieldCheck, UserPlus } from "lucide-react";
import { FormEvent, useState } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/lib/auth";

export default function RegisterPage() {
  const { user, register } = useAuth();
  const navigate = useNavigate();

  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  if (user) {
    return <Navigate to="/upload" replace />;
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (password !== confirmPassword) {
      setError("两次输入的密码不一致");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await register({
        username: username.trim(),
        email: email.trim(),
        password,
        display_name: displayName.trim() || undefined,
      });
      navigate("/upload", { replace: true });
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
          <h1 className="text-2xl font-bold text-slate-900">创建账号</h1>
          <p className="mt-2 text-sm text-slate-500">注册后即可使用流量上传与模型分析功能</p>
        </div>

        <Card className="border-0 shadow-xl shadow-slate-200/60">
          <CardHeader title="新用户注册" description="用户名为 3–32 位字母、数字或下划线" />
          <CardBody>
            <form className="space-y-4" onSubmit={(e) => void handleSubmit(e)}>
              <Input
                label="用户名"
                name="username"
                autoComplete="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="例如 analyst01"
                pattern="[A-Za-z0-9_]{3,32}"
                required
              />
              <Input
                label="邮箱"
                name="email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                required
              />
              <Input
                label="显示名称（可选）"
                name="displayName"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="在界面中展示的名称"
              />
              <Input
                label="密码"
                name="password"
                type="password"
                autoComplete="new-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="至少 6 位"
                minLength={6}
                required
              />
              <Input
                label="确认密码"
                name="confirmPassword"
                type="password"
                autoComplete="new-password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="再次输入密码"
                minLength={6}
                required
              />

              {error ? (
                <p className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</p>
              ) : null}

              <Button type="submit" className="w-full" disabled={submitting}>
                {submitting ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <UserPlus className="h-4 w-4" />
                )}
                注册并登录
              </Button>
            </form>

            <p className="mt-6 text-center text-sm text-slate-500">
              已有账号？{" "}
              <Link to="/login" className="font-medium text-brand-600 hover:text-brand-700">
                返回登录
              </Link>
            </p>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
