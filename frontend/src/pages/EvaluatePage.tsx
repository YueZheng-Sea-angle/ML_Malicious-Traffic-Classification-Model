import { FlaskConical, Loader2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { Badge, labelTone } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardBody, CardHeader } from "@/components/ui/card";
import { api, type EvalJob, type ModelInfo } from "@/lib/api";
import { formatPercent } from "@/lib/utils";

const TOOL_ZH: Record<string, string> = {
  "openvpn-udp": "OpenVPN（UDP）",
  "psiphon-tls": "Psiphon（TLS）",
  v2ray: "V2Ray",
  clash: "Clash",
  lantern: "Lantern",
  "openvpn-tls": "OpenVPN（TLS）",
  firefox: "Firefox 直连",
  "psiphon-tcp": "Psiphon（TCP）",
  "wireguard-udp": "WireGuard（UDP）",
  shadowsocks: "Shadowsocks",
  netch: "Netch",
};

const VARIANT_LABEL: Record<string, string> = {
  A: "A · 统计基线（40 维统计）",
  B: "B · 统计 + 载荷字节 n-gram",
};

export default function EvaluatePage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [ratio, setRatio] = useState(20); // 训练集比例 %
  const [perClass, setPerClass] = useState(10);
  const [job, setJob] = useState<EvalJob | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refreshModels = useCallback(async () => {
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
    void refreshModels();
  }, [refreshModels]);

  const activeModel = models.find((m) => m.model_id === activeId) ?? null;

  const start = useCallback(async () => {
    if (!activeId) return;
    setLoading(true);
    setError(null);
    setJob(null);
    try {
      const created = await api.startEvaluate({
        model_id: activeId,
        train_ratio: ratio / 100,
        per_class_files: perClass,
      });
      let current = created;
      const deadline = Date.now() + 20 * 60 * 1000;
      while (current.status === "running" && Date.now() < deadline) {
        await new Promise((r) => setTimeout(r, 2000));
        current = await api.getEvaluate(current.job_id);
      }
      if (current.status === "running") {
        throw new Error("测试超时未完成，请尝试降低每类文件数或加大训练集比例");
      }
      setJob(current);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [activeId, ratio, perClass]);

  return (
    <div className="space-y-6">
      {error ? (
        <p className="msg-error">{error}</p>
      ) : null}

      <div className="grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-2">
          <CardHeader
            title="模型性能测试"
            description="用 real_data 随机文件级划分，重训当前激活模型并测试；结果自动写入「模型管理」对应卡片的准确率/Macro-F1"
          />
          <CardBody className="space-y-4">
            <div>
              <p className="mb-1 text-xs text-slate-500">当前被测模型（模型管理选中的激活模型）</p>
              {activeModel ? (
                <div className="flex items-center gap-2">
                  <Badge tone={activeModel.model_id.includes("ngram") ? "warning" : "info"}>
                    {activeModel.model_id.includes("ngram") ? "B · n-gram 增强" : "A · 基线"}
                  </Badge>
                  <span className="truncate font-mono text-xs text-slate-700 dark:text-slate-300">
                    {activeModel.model_id}
                  </span>
                </div>
              ) : (
                <p className="text-sm text-slate-400">暂无激活模型，请先到「模型管理」激活。</p>
              )}
            </div>

            <label className="block">
              <span className="mb-1 block text-xs text-slate-500">训练集比例（%）· 默认 20</span>
              <input
                type="number"
                min={5}
                max={95}
                step={5}
                value={ratio}
                onChange={(e) => setRatio(Number(e.target.value))}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-950 dark:text-slate-100"
              />
              <span className="mt-1 block text-xs text-slate-400 dark:text-slate-500">
                按文件分层随机划分：训练 {ratio}% / 测试 {100 - ratio}%
              </span>
            </label>

            <label className="block">
              <span className="mb-1 block text-xs text-slate-500">每类文件数上限</span>
              <input
                type="number"
                min={1}
                max={100}
                value={perClass}
                onChange={(e) => setPerClass(Number(e.target.value))}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-950 dark:text-slate-100"
              />
              <span className="mt-1 block text-xs text-slate-400 dark:text-slate-500">
                每类从 real_data 随机抽取的文件数（避免全量解析过慢）
              </span>
            </label>

            <Button onClick={() => void start()} disabled={loading || !activeModel} className="w-full">
              {loading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> 测试运行中…
                </>
              ) : (
                <>开始测试</>
              )}
            </Button>
          </CardBody>
        </Card>

        <Card className="lg:col-span-3">
          <CardHeader
            title="测试结果"
            description={
              job
                ? `任务 ${job.job_id} · ${VARIANT_LABEL[job.variant] ?? job.variant}`
                : "选择参数后点击“开始测试”"
            }
          />
          <CardBody className="space-y-5">
            {!job ? (
              <p className="flex items-center gap-2 text-sm text-slate-500">
                <FlaskConical className="h-4 w-4" />
                尚未运行。测试会使用 real_data 按比例划分训练/测试集并重新训练所选变体。
              </p>
            ) : job.status === "running" ? (
              <p className="flex items-center gap-2 text-sm text-slate-500">
                <Loader2 className="h-4 w-4 animate-spin" />
                正在解析流量并训练 LightGBM（{job.variant} 变体）…
              </p>
            ) : job.status === "failed" ? (
              <p className="msg-error">{job.error ?? "测试失败"}</p>
            ) : (
              job.result && <ResultView job={job} />
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

function ResultView({ job }: { job: EvalJob }) {
  const r = job.result!;
  const recalls = Object.entries(r.test_metrics.per_class_recall).sort((a, b) => b[1] - a[1]);
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
        <Badge tone="neutral">训练 {r.n_train_files} 文件 / {r.n_train_flows} 流</Badge>
        <Badge tone="neutral">测试 {r.n_test_files} 文件 / {r.n_test_flows} 流</Badge>
        <Badge tone="neutral">耗时 {job.elapsed_ms ?? 0} ms</Badge>
        {r.ngram_vocab_size > 0 ? (
          <Badge tone="neutral">n-gram 词表 {r.ngram_vocab_size}</Badge>
        ) : null}
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <ResultStat label="流级准确率" value={r.test_metrics.accuracy} />
        <ResultStat label="流级 Macro-F1" value={r.test_metrics.macro_f1} />
        <ResultStat label="文件级命中率" value={r.file_metrics.hit_rate} sub={`${r.file_metrics.n_files} 文件`} />
      </div>

      <section>
        <h4 className="mb-2 text-sm font-medium text-slate-700 dark:text-slate-300">
          逐类召回（11 类代理/隧道工具）
        </h4>
        <div className="data-table-wrap">
          <table className="w-full text-left text-xs">
            <thead>
              <tr>
                <th className="px-3 py-2 font-medium">类别</th>
                <th className="px-3 py-2 text-right font-medium">召回率</th>
              </tr>
            </thead>
            <tbody>
              {recalls.map(([name, recall]) => (
                <tr key={name}>
                  <td className="px-3 py-2">
                    <Badge tone={labelTone(name)}>{TOOL_ZH[name] ?? name}</Badge>
                  </td>
                  <td className="px-3 py-2 text-right tabular-nums text-slate-600">
                    {formatPercent(recall)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function ResultStat({ label, value, sub }: { label: string; value: number; sub?: string }) {
  return (
    <Card>
      <CardBody>
        <p className="text-xs text-slate-500">{label}</p>
        <p className="stat-value mt-1">{formatPercent(value)}</p>
        {sub ? <p className="text-xs text-slate-400">{sub}</p> : null}
      </CardBody>
    </Card>
  );
}
