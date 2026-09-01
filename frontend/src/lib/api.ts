/**
 * 后端接口封装。类型定义须与 backend/app/schemas.py 保持一致。
 */

const BASE = import.meta.env.VITE_API_BASE ?? "/api";

export interface Health {
  status: string;
  app_name: string;
  version: string;
  inference_mode: "model" | "heuristic" | "unavailable";
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
  malicious_score: number;
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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail?.detail ?? `请求失败（${response.status}）`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health"),

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
