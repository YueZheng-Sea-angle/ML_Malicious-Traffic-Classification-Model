import { CheckCircle2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { api, type ModelInfo } from "@/lib/api";
import { formatPercent, formatTime } from "@/lib/utils";

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const data = await api.listModels();
      setModels(data.items);
      setActiveId(data.active_model_id);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const activate = async (modelId: string) => {
    setPending(modelId);
    try {
      await api.activateModel(modelId);
      await refresh();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(null);
    }
  };

  return (
    <div className="space-y-6">
      {error ? (
        <p className="rounded-lg bg-rose-50 px-4 py-3 text-sm text-rose-700">{error}</p>
      ) : null}

      <Card>
        <CardHeader
          title="模型仓库"
          description="权重文件来自 artifacts/models，训练命令：python -m ml.train --synthetic"
        />
        <CardBody className="space-y-3">
          {models.map((model) => (
            <div
              key={model.model_id}
              className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-200 px-4 py-3"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <p className="font-medium text-slate-800">{model.name}</p>
                  {model.model_id === activeId ? (
                    <Badge tone="success">
                      <CheckCircle2 className="mr-1 h-3 w-3" />
                      使用中
                    </Badge>
                  ) : null}
                </div>
                <p className="mt-1 text-xs text-slate-500">
                  {model.model_id} · {model.framework} · v{model.version} · 训练于{" "}
                  {formatTime(model.trained_at)}
                </p>
                {model.selected_features.length > 0 ? (
                  <p className="mt-1 truncate text-xs text-slate-400">
                    自适应保留特征 {model.selected_features.length} 维：
                    {model.selected_features.slice(0, 6).join("、")}…
                  </p>
                ) : null}
              </div>

              <div className="flex items-center gap-4">
                <Metric label="准确率" value={model.accuracy} />
                <Metric label="Macro-F1" value={model.macro_f1} />
                <Button
                  variant={model.model_id === activeId ? "outline" : "primary"}
                  size="sm"
                  disabled={model.model_id === activeId || pending === model.model_id}
                  onClick={() => void activate(model.model_id)}
                >
                  {model.model_id === activeId ? "已激活" : "激活"}
                </Button>
              </div>
            </div>
          ))}
          {models.length === 0 ? (
            <p className="text-sm text-slate-500">尚无模型，请先执行训练脚本。</p>
          ) : null}
        </CardBody>
      </Card>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="text-right">
      <p className="text-xs text-slate-400">{label}</p>
      <p className="text-sm font-semibold tabular-nums text-slate-700">
        {value == null ? "—" : formatPercent(value)}
      </p>
    </div>
  );
}
