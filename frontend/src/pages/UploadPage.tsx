import { FileUp, Loader2 } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";

import { Badge, labelTone } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { api, waitForTask, type Task } from "@/lib/api";
import { cn, formatBytes, formatPercent } from "@/lib/utils";

type Phase = "idle" | "uploading" | "analyzing" | "done" | "error";

export default function UploadPage() {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [dragging, setDragging] = useState(false);
  const [message, setMessage] = useState("");
  const [task, setTask] = useState<Task | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setPhase("uploading");
    setMessage(`正在上传 ${file.name}（${formatBytes(file.size)}）`);
    setTask(null);
    try {
      const uploaded = await api.uploadFile(file);
      setPhase("analyzing");
      setMessage("已上传，正在提取特征并推理…");
      const created = await api.createTask(uploaded.file_id);
      const finished = await waitForTask(created.task_id);
      setTask(finished);
      setPhase(finished.status === "succeeded" ? "done" : "error");
      setMessage(finished.status === "succeeded" ? "分析完成" : finished.error ?? "分析失败");
    } catch (e) {
      setPhase("error");
      setMessage((e as Error).message);
    }
  }, []);

  const busy = phase === "uploading" || phase === "analyzing";

  return (
    <div className="grid gap-6 lg:grid-cols-5">
      <Card className="lg:col-span-3">
        <CardHeader
          title="上传流量文件"
          description="支持 .pcap / .pcapng / .cap，单文件不超过 200 MB"
        />
        <CardBody>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragging(false);
              const file = e.dataTransfer.files?.[0];
              if (file) void handleFile(file);
            }}
            className={cn(
              "flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors",
              dragging
                ? "border-brand-500 bg-brand-50 dark:bg-brand-950/40"
                : "border-slate-300 bg-slate-50 dark:border-slate-600 dark:bg-slate-800/40",
            )}
          >
            {busy ? (
              <Loader2 className="h-10 w-10 animate-spin text-brand-500" />
            ) : (
              <FileUp className="h-10 w-10 text-slate-400" />
            )}
            <p className="text-sm text-slate-600 dark:text-slate-400">
              {busy ? message : "将流量文件拖拽到此处，或点击下方按钮选择"}
            </p>
            <input
              ref={inputRef}
              type="file"
              accept=".pcap,.pcapng,.cap"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) void handleFile(file);
                e.target.value = "";
              }}
            />
            <Button disabled={busy} onClick={() => inputRef.current?.click()}>
              选择文件并分析
            </Button>
          </div>

          {phase === "error" ? (
            <p className="msg-error mt-4">{message}</p>
          ) : null}
        </CardBody>
      </Card>

      <Card className="lg:col-span-2">
        <CardHeader
          title="本次分析结论"
          description={task ? `任务 ${task.task_id}` : "上传后自动展示"}
          action={
            task ? (
              <Button variant="outline" size="sm" onClick={() => navigate("/results")}>
                查看全部
              </Button>
            ) : null
          }
        />
        <CardBody className="space-y-4">
          {!task?.result ? (
            <p className="text-sm text-slate-500">暂无结果。</p>
          ) : (
            <>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-xs text-slate-500">文件级判定（T1 工具识别）</p>
                  <p className="stat-value">{task.result.label_zh}</p>
                </div>
                <Badge tone={labelTone(task.result.label)}>
                  置信度 {formatPercent(task.result.confidence)}
                </Badge>
              </div>

              <div className="space-y-2">
                <p className="text-xs font-medium text-slate-500">各工具类别概率</p>
                {Object.entries(task.result.probabilities)
                  .sort((a, b) => b[1] - a[1])
                  .map(([name, value]) => (
                    <div key={name} className="flex items-center gap-3">
                      <span className="w-24 shrink-0 text-xs text-slate-600">{name}</span>
                      <Progress value={value} className="flex-1" />
                      <span className="w-12 text-right text-xs tabular-nums text-slate-500">
                        {formatPercent(value)}
                      </span>
                    </div>
                  ))}
              </div>

              <p className="text-xs text-slate-400">
                共解析 {task.result.flow_count} 条流 · 耗时 {task.elapsed_ms ?? 0} ms · 推理模式{" "}
                {task.result.mode}
              </p>
            </>
          )}
        </CardBody>
      </Card>
    </div>
  );
}
