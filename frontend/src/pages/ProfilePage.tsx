import { Check, KeyRound, Loader2, Save, UserCircle2 } from "lucide-react";
import { FormEvent, useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useAuth } from "@/lib/auth";
import { api } from "@/lib/api";
import { formatTime } from "@/lib/utils";

export default function ProfilePage() {
  const { user, refreshUser } = useAuth();

  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [profileMessage, setProfileMessage] = useState<string | null>(null);
  const [profileError, setProfileError] = useState<string | null>(null);
  const [profileSaving, setProfileSaving] = useState(false);

  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [passwordMessage, setPasswordMessage] = useState<string | null>(null);
  const [passwordError, setPasswordError] = useState<string | null>(null);
  const [passwordSaving, setPasswordSaving] = useState(false);

  useEffect(() => {
    if (user) {
      setDisplayName(user.display_name);
      setEmail(user.email);
    }
  }, [user]);

  async function handleProfileSubmit(event: FormEvent) {
    event.preventDefault();
    setProfileSaving(true);
    setProfileMessage(null);
    setProfileError(null);
    try {
      await api.updateProfile({ display_name: displayName.trim(), email: email.trim() });
      await refreshUser();
      setProfileMessage("个人资料已保存");
    } catch (e) {
      setProfileError((e as Error).message);
    } finally {
      setProfileSaving(false);
    }
  }

  async function handlePasswordSubmit(event: FormEvent) {
    event.preventDefault();
    if (newPassword !== confirmPassword) {
      setPasswordError("两次输入的新密码不一致");
      return;
    }
    setPasswordSaving(true);
    setPasswordMessage(null);
    setPasswordError(null);
    try {
      await api.changePassword({ current_password: currentPassword, new_password: newPassword });
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setPasswordMessage("密码修改成功");
    } catch (e) {
      setPasswordError((e as Error).message);
    } finally {
      setPasswordSaving(false);
    }
  }

  if (!user) {
    return null;
  }

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <Card className="lg:col-span-2">
        <CardHeader title="账号概览" description="当前登录用户信息" />
        <CardBody className="space-y-5">
          <div className="flex items-center gap-4">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-500 to-brand-700 text-2xl font-bold text-white">
              {user.display_name.slice(0, 1).toUpperCase()}
            </div>
            <div>
              <p className="text-lg font-semibold text-slate-900">{user.display_name}</p>
              <p className="text-sm text-slate-500">@{user.username}</p>
            </div>
          </div>

          <dl className="space-y-3 text-sm">
            <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
              <dt className="text-slate-500">角色</dt>
              <dd>
                <Badge tone="info">{user.role}</Badge>
              </dd>
            </div>
            <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
              <dt className="text-slate-500">注册时间</dt>
              <dd className="text-slate-700">{formatTime(user.created_at)}</dd>
            </div>
            <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2">
              <dt className="text-slate-500">最近更新</dt>
              <dd className="text-slate-700">{formatTime(user.updated_at)}</dd>
            </div>
          </dl>
        </CardBody>
      </Card>

      <div className="space-y-6 lg:col-span-3">
        <Card>
          <CardHeader
            title="编辑资料"
            description="修改显示名称与联系邮箱"
            action={<UserCircle2 className="h-5 w-5 text-slate-400" />}
          />
          <CardBody>
            <form className="space-y-4" onSubmit={(e) => void handleProfileSubmit(e)}>
              <Input
                label="显示名称"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                required
              />
              <Input
                label="邮箱"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />

              {profileError ? (
                <p className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">{profileError}</p>
              ) : null}
              {profileMessage ? (
                <p className="flex items-center gap-2 rounded-lg bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                  <Check className="h-4 w-4" />
                  {profileMessage}
                </p>
              ) : null}

              <Button type="submit" disabled={profileSaving}>
                {profileSaving ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Save className="h-4 w-4" />
                )}
                保存资料
              </Button>
            </form>
          </CardBody>
        </Card>

        <Card>
          <CardHeader
            title="修改密码"
            description="修改成功后当前会话仍然有效"
            action={<KeyRound className="h-5 w-5 text-slate-400" />}
          />
          <CardBody>
            <form className="space-y-4" onSubmit={(e) => void handlePasswordSubmit(e)}>
              <Input
                label="当前密码"
                type="password"
                autoComplete="current-password"
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
              />
              <Input
                label="新密码"
                type="password"
                autoComplete="new-password"
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                minLength={6}
                hint="至少 6 位字符"
                required
              />
              <Input
                label="确认新密码"
                type="password"
                autoComplete="new-password"
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                minLength={6}
                required
              />

              {passwordError ? (
                <p className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">{passwordError}</p>
              ) : null}
              {passwordMessage ? (
                <p className="flex items-center gap-2 rounded-lg bg-emerald-50 px-4 py-3 text-sm text-emerald-700">
                  <Check className="h-4 w-4" />
                  {passwordMessage}
                </p>
              ) : null}

              <Button type="submit" variant="outline" disabled={passwordSaving}>
                {passwordSaving ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <KeyRound className="h-4 w-4" />
                )}
                更新密码
              </Button>
            </form>
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
