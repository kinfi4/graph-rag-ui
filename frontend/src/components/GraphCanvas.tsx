import { forceCollide, forceX, forceY } from "d3-force";
import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods } from "react-force-graph-2d";

import type { GraphLink, GraphNode } from "../api";
import { typeColor } from "../theme";

type Node = GraphNode & { x: number; y: number };
type Link = Omit<GraphLink, "source" | "target"> & { source: Node | string; target: Node | string };

interface Props {
  nodes: GraphNode[];
  links: GraphLink[];
  step: number;
  highlight: Set<string> | null;
  selected: string | null;
  onHover: (node: GraphNode | null) => void;
  onSelect: (node: GraphNode | null) => void;
}

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 600, height: 400 });
  useEffect(() => {
    if (!ref.current) return;
    const observer = new ResizeObserver(([entry]) => {
      const { width, height } = entry.contentRect;
      setSize({ width: Math.max(200, width), height: Math.max(200, height) });
    });
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);
  return { ref, ...size };
}

const activationAt = (node: GraphNode, step: number) =>
  node.activation[Math.min(step, node.activation.length - 1)] ?? 0;

const endpoint = (end: Node | string) => (typeof end === "string" ? end : end.id);

export default function GraphCanvas({
  nodes,
  links,
  step,
  highlight,
  selected,
  onHover,
  onSelect,
}: Props) {
  const { ref, width, height } = useSize<HTMLDivElement>();
  const graphRef = useRef<ForceGraphMethods<Node, Link>>();
  const [hovered, setHovered] = useState<string | null>(null);
  const fitted = useRef(false);

  // force-graph mutates its input; copy once per answer so re-renders keep the layout
  const data = useMemo(() => {
    fitted.current = false;
    return {
      nodes: nodes.map((n) => ({ ...n })) as Node[],
      links: links.map((l) => ({ ...l })) as Link[],
    };
  }, [nodes, links]);

  const byId = useMemo(() => new Map(data.nodes.map((n) => [n.id, n])), [data]);
  const maxActivation = useMemo(
    () => Math.max(1e-6, ...nodes.flatMap((n) => n.activation)),
    [nodes],
  );

  useEffect(() => {
    const graph = graphRef.current;
    if (!graph) return;
    graph.d3Force("charge")?.strength(-220);
    graph.d3Force("link")?.distance(58);
    // gravity keeps disconnected components close, so zoom-to-fit doesn't shrink the main one
    graph.d3Force("x", forceX(0).strength(0.07));
    graph.d3Force("y", forceY(0).strength(0.07));
    graph.d3Force("collide", forceCollide(16));
    graph.d3ReheatSimulation();
  }, [data]);

  const radius = (node: GraphNode) => {
    const act = activationAt(node, step);
    if (act <= 0) return 2.6;
    return 3.6 + 8.5 * Math.sqrt(Math.min(1, act / maxActivation));
  };

  const isActive = (id: string) => {
    const node = byId.get(id);
    return node ? activationAt(node, step) > 0 : false;
  };

  const paintNode = (node: Node, ctx: CanvasRenderingContext2D, scale: number) => {
    const active = activationAt(node, step) > 0;
    const dimmed = highlight !== null && !highlight.has(node.id);
    const r = radius(node);
    const glowing = node.evidence && active && !dimmed;

    ctx.save();
    ctx.globalAlpha = dimmed ? 0.12 : active ? 1 : 0.2;
    if (glowing) {
      ctx.shadowColor = "rgba(255,255,255,0.9)";
      ctx.shadowBlur = 16;
    }
    ctx.beginPath();
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
    ctx.fillStyle = typeColor(node.type);
    ctx.fill();
    ctx.shadowBlur = 0;

    if (glowing) {
      ctx.lineWidth = 1.3;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
    }
    if (node.is_seed) {
      ctx.setLineDash([2.4, 1.8]);
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = "#f0efec";
      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 3, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (selected === node.id) {
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = "#22d3ee";
      ctx.beginPath();
      ctx.arc(node.x, node.y, r + 5.5, 0, 2 * Math.PI);
      ctx.stroke();
    }

    const showLabel =
      !dimmed &&
      (node.is_seed ||
        (node.evidence && active) ||
        hovered === node.id ||
        selected === node.id ||
        highlight?.has(node.id) ||
        (scale > 2.4 && active));
    if (showLabel) {
      const fontSize = 11 / scale;
      const label = node.id.length > 32 ? `${node.id.slice(0, 31)}…` : node.id;
      ctx.font = `${node.evidence ? 600 : 500} ${fontSize}px Inter, sans-serif`;
      const textWidth = ctx.measureText(label).width;
      const pad = 3 / scale;
      const top = node.y + r + 4 / scale;
      ctx.fillStyle = "rgba(7,7,13,0.82)";
      ctx.beginPath();
      ctx.roundRect(node.x - textWidth / 2 - pad, top, textWidth + 2 * pad, fontSize + 2 * pad, pad);
      ctx.fill();
      ctx.fillStyle = node.evidence ? "#ffffff" : "#c3c2b7";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(label, node.x, top + pad);
    }
    ctx.restore();
  };

  const linkState = (link: Link) => {
    const source = endpoint(link.source);
    const target = endpoint(link.target);
    const live = isActive(source) && isActive(target);
    const dimmed = highlight !== null && !(highlight.has(source) && highlight.has(target));
    return { live, dimmed };
  };

  const linkColor = (link: Link) => {
    const { live, dimmed } = linkState(link);
    if (dimmed || !live) return "rgba(180,182,212,0.06)";
    if (link.evidence) return "rgba(255,255,255,0.92)";
    if (link.on_path) return "rgba(167,139,250,0.6)";
    return "rgba(180,182,212,0.18)";
  };

  const paintLinkLabel = (link: Link, ctx: CanvasRenderingContext2D, scale: number) => {
    const { live, dimmed } = linkState(link);
    const source = link.source as Node;
    const target = link.target as Node;
    const focused = hovered && (source.id === hovered || target.id === hovered);
    if (!live || dimmed || !(link.evidence || focused) || typeof source !== "object") return;
    const label = link.relation.length > 36 ? `${link.relation.slice(0, 35)}…` : link.relation;
    const fontSize = 9 / scale;
    ctx.save();
    ctx.font = `500 ${fontSize}px "JetBrains Mono", monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const x = (source.x + target.x) / 2;
    const y = (source.y + target.y) / 2;
    const w = ctx.measureText(label).width + 6 / scale;
    ctx.fillStyle = "rgba(14,15,26,0.9)";
    ctx.fillRect(x - w / 2, y - fontSize / 2 - 2 / scale, w, fontSize + 4 / scale);
    ctx.fillStyle = link.evidence ? "#d9f99d" : "#b4b6d4";
    ctx.fillText(label, x, y);
    ctx.restore();
  };

  return (
    <div ref={ref} className="absolute inset-0">
      <ForceGraph2D<Node, Link>
        ref={graphRef}
        width={width}
        height={height}
        graphData={data}
        backgroundColor="rgba(0,0,0,0)"
        autoPauseRedraw={false}
        nodeId="id"
        nodeLabel={() => ""}
        nodeCanvasObject={paintNode}
        nodeCanvasObjectMode={() => "replace"}
        nodePointerAreaPaint={(node, color, ctx) => {
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(node.x, node.y, radius(node) + 4, 0, 2 * Math.PI);
          ctx.fill();
        }}
        linkColor={linkColor}
        linkWidth={(link) => (link.evidence ? 2.2 : link.on_path ? 1.1 : 0.6)}
        linkDirectionalArrowLength={(link) => (linkState(link).live ? 3.2 : 0)}
        linkDirectionalArrowRelPos={0.92}
        linkDirectionalArrowColor={linkColor}
        linkDirectionalParticles={(link) => (link.evidence && linkState(link).live ? 3 : 0)}
        linkDirectionalParticleWidth={2.6}
        linkDirectionalParticleSpeed={0.007}
        linkDirectionalParticleColor={() => "#d9f99d"}
        linkCanvasObjectMode={() => "after"}
        linkCanvasObject={paintLinkLabel}
        cooldownTicks={150}
        onEngineStop={() => {
          if (!fitted.current) {
            graphRef.current?.zoomToFit(500, 50);
            fitted.current = true;
          }
        }}
        onNodeHover={(node) => {
          setHovered(node?.id ?? null);
          onHover(node ?? null);
        }}
        onNodeClick={(node) => onSelect(node)}
        onBackgroundClick={() => onSelect(null)}
      />
    </div>
  );
}
