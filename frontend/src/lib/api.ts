/**
 * 后端接口封装。类型定义须与 backend/app/schemas.py 保持一致。
 */

const BASE = import.meta.env.VITE_API_BASE ?? "/api";
const TOKEN_KEY = "malflow_token";

export function getAuthToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearAuthToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

export interface User {
  user_id: string;
  username: string;
  email: string;
  display_name: string;
  role: string;
  created_at: string;
  updated_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}

export interface Health {
  status: string;
  app_name: string;
  version: string;
  inference_mode: "model" | "unavailable";
  torch_available: boolean;
}

export interface UploadedFile {
  file_id: string;
  filename: string;
  size_bytes: number;
  uploaded_at: string;
}

export interface FlowResult {
  flow_id: string;
  label: string;
  label_zh: string;
  confidence: number;
  probabilities: Record<string, number>;
  meta: Record<string, string | number>;
}

export interface FeatureContribution {
  name: string;
  weight: number;
  value: number;
}

export interface TaskResult {
  label: string;
  label_zh: string;
  confidence: number;
  probabilities: Record<string, number>;
  flow_count: number;
  flows: FlowResult[];
  top_features: FeatureContribution[];
  mode: string;
}

export interface Task {
  task_id: string;
  file_id: string;
  filename: string;
  status: "pending" | "running" | "succeeded" | "failed";
  model_id: string | null;
  created_at: string;
  finished_at: string | null;
  elapsed_ms: number | null;
  error: string | null;
  result: TaskResult | null;
}

export interface ModelInfo {
  model_id: string;
  name: string;
  version: string;
  framework: string;
  is_active: boolean;
  accuracy: number | null;
  macro_f1: number | null;
  trained_at: string | null;
  selected_features: string[];
  description: string;
}

export interface Stats {
  total_tasks: number;
  succeeded: number;
  failed: number;
  running: number;
  label_distribution: Record<string, number>;
  average_elapsed_ms: number;
}

export interface EvalJob {
  job_id: string;
  model_id: string;
  variant: "A" | "B";
  train_ratio: number;
  per_class_files: number;
  status: "running" | "succeeded" | "failed";
  error: string | null;
  elapsed_ms: number | null;
  result: {
    variant: "A" | "B";
    n_train_files: number;
    n_test_files: number;
    n_train_flows: number;
    n_test_flows: number;
    ngram_vocab_size: number;
    test_metrics: {
      accuracy: number;
      macro_f1: number;
      per_class_recall: Record<string, number>;
    };
    file_metrics: { n_files: number; hit_rate: number };
  } | null;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers);
  const token = getAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(`${BASE}${path}`, { ...init, headers });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message =
      typeof detail?.detail === "string"
        ? detail.detail
        : Array.isArray(detail?.detail)
          ? detail.detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join("；")
          : `请求失败（${response.status}）`;
    throw new Error(message);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

  register: (payload: {
    username: string;
    email: string;
    password: string;
    display_name?: string;
  }) =>
    request<AuthResponse>("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  login: (payload: { username: string; password: string }) =>
    request<AuthResponse>("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  logout: () => request<{ ok: boolean }>("/auth/logout", { method: "POST" }),

  me: () => request<User>("/auth/me"),

  updateProfile: (payload: { display_name?: string; email?: string }) =>
    request<User>("/auth/profile", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  changePassword: (payload: { current_password: string; new_password: string }) =>
    request<{ ok: boolean }>("/auth/change-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  uploadFile: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<UploadedFile>("/traffic/upload", { method: "POST", body: form });
  },

  listFiles: () => request<UploadedFile[]>("/traffic/files"),

  createTask: (fileId: string, maxFlows = 64) =>
    request<Task>("/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_id: fileId, max_flows: maxFlows }),
    }),

  getTask: (taskId: string) => request<Task>(`/tasks/${taskId}`),

  listTasks: (limit = 50) =>
    request<{ total: number; items: Task[] }>(`/tasks?limit=${limit}`),

  stats: () => request<Stats>("/tasks/stats"),

  listModels: () =>
    request<{ total: number; active_model_id: string | null; items: ModelInfo[] }>("/models"),

  activateModel: (modelId: string) =>
    request<{ activated: string }>(`/models/${modelId}/activate`, { method: "POST" }),

  startEvaluate: (payload: { model_id?: string; train_ratio: number; per_class_files: number }) =>
    request<EvalJob>("/models/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  getEvaluate: (jobId: string) => request<EvalJob>(`/models/evaluate/${jobId}`),
};

/** 轮询任务直到完成，用于上传后自动展示结果。 */
export async function waitForTask(taskId: string, timeoutMs = 60000): Promise<Task> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const task = await api.getTask(taskId);
    if (task.status === "succeeded" || task.status === "failed") return task;
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
  throw new Error("任务超时未完成");
}
