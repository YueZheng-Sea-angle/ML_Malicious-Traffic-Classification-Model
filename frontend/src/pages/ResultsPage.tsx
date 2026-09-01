import { RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { Badge, labelTone, statusTone } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { api, type Stats, type Task } from "@/lib/api";
import { cn, formatPercent, formatTime } from "@/lib/utils";

export default function ResultsPage() {
  const [tasks, setTasks] = useState<Task[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [selected, setSelected] = useState<Task | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [taskList, statistics] = await Promise.all([api.listTasks(), api.stats()]);
      setTasks(taskList.items);
      setStats(statistics);
      setSelected((current) =>
        current ? taskList.items.find((t) => t.task_id === current.task_id) ?? null : taskList.items[0] ?? null,
      );
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-4">
        <StatCard label="任务总数" value={stats?.total_tasks ?? 0} />
        <StatCard label="成功" value={stats?.succeeded ?? 0} />
        <StatCard label="进行中" value={stats?.running ?? 0} />
        <StatCard label="平均耗时" value={`${stats?.average_elapsed_ms ?? 0} ms`} />
      </div>

      {error ? (
        <p className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</p>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-2">
          <CardHeader
            title="任务列表"
            description="点击查看流级明细"
            action={
              <Button variant="outline" size="sm" onClick={() => void refresh()} disabled={loading}>
                <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} />
                刷新
              </Button>
            }
          />
          <CardBody className="max-h-[520px] space-y-2 overflow-y-auto">
            {tasks.length === 0 ? (
              <p className="text-sm text-slate-500">暂无任务，请先在「流量上传」页提交文件。</p>
            ) : (
              tasks.map((task) => (
                <button
                  key={task.task_id}
                  onClick={() => setSelected(task)}
                  className={cn(
                    "flex w-full items-center justify-between rounded-lg border px-3 py-2 text-left transition-colors",
                    selected?.task_id === task.task_id
                      ? "border-brand-500 bg-brand-50"
                      : "border-slate-200 hover:bg-slate-50",
                  )}
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium text-slate-800">{task.filename}</p>
                    <p className="text-xs text-slate-400">{formatTime(task.created_at)}</p>
                  </div>
                  <div className="flex shrink-0 items-center gap-2">
                    {task.result ? (
                      <Badge tone={labelTone(task.result.label)}>{task.result.label_zh}</Badge>
                    ) : null}
                    <Badge tone={statusTone(task.status)}>{task.status}</Badge>
                  </div>
                </button>
              ))
            )}
          </CardBody>
        </Card>

        <Card className="lg:col-span-3">
          <CardHeader
            title="分析详情"
            description={selected ? `${selected.filename} · ${selected.task_id}` : "未选择任务"}
          />
          <CardBody className="space-y-5">
            {!selected?.result ? (
              <p className="text-sm text-slate-500">
                {selected?.error ?? "选择左侧任务以查看流级分类结果与关键特征。"}
              </p>
            ) : (
              <>
                <section>
                  <h4 className="mb-2 text-sm font-medium text-slate-700">关键特征贡献</h4>
                  <div className="space-y-2">
                    {selected.result.top_features.map((feature) => (
                      <div key={feature.name} className="flex items-center gap-3">
                        <span className="w-40 shrink-0 truncate text-xs text-slate-600">
                          {feature.name}
                        </span>
                        <Progress value={feature.weight} className="flex-1" />
                        <span className="w-16 text-right text-xs tabular-nums text-slate-500">
                          {feature.value}
                        </span>
                      </div>
                    ))}
                  </div>
                </section>

                <section>
                  <h4 className="mb-2 text-sm font-medium text-slate-700">
                    流级结果（{selected.result.flow_count} 条）
                  </h4>
                  <div className="max-h-72 overflow-y-auto rounded-lg border border-slate-200">
                    <table className="w-full text-left text-xs">
                      <thead className="sticky top-0 bg-slate-50 text-slate-500">
                        <tr>
                          <th className="px-3 py-2 font-medium">流标识</th>
                          <th className="px-3 py-2 font-medium">包数</th>
                          <th className="px-3 py-2 font-medium">类别</th>
                          <th className="px-3 py-2 text-right font-medium">置信度</th>
                        </tr>
                      </thead>
                      <tbody>
                        {selected.result.flows.map((flow) => (
                          <tr key={flow.flow_id} className="border-t border-slate-100">
                            <td className="max-w-[220px] truncate px-3 py-2 font-mono text-[11px] text-slate-600">
                              {flow.flow_id}
                            </td>
                            <td className="px-3 py-2 tabular-nums text-slate-500">
                              {String(flow.meta?.packets ?? "—")}
                            </td>
                            <td className="px-3 py-2">
                              <Badge tone={labelTone(flow.label)}>{flow.label_zh}</Badge>
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-slate-600">
                              {formatPercent(flow.confidence)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              </>
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <Card>
      <CardBody>
        <p className="text-xs text-slate-500">{label}</p>
        <p className="stat-value mt-1">{value}</p>
      </CardBody>
    </Card>
  );
}
