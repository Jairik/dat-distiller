/**
 * Dataset Version tree: lineage from parent_id, branches indented with
 * guide lines, the path to the selected Version highlighted as the current
 * branch, and each row's Provenance breakdown inline.
 */

import { motion } from 'motion/react'
import { Badge } from '@/components/ui/badge'
import { ProvenanceBar } from '@/components/dataset/provenance-bar'
import { fmtNumber, timeAgo } from '@/lib/format'
import { cn } from '@/lib/utils'
import type { DatasetVersion } from '@/lib/projects'

interface TreeNode extends DatasetVersion {
  children: TreeNode[]
}

function buildTree(versions: DatasetVersion[]): TreeNode[] {
  const nodes = new Map<string, TreeNode>()
  for (const version of versions) nodes.set(version.id, { ...version, children: [] })
  const roots: TreeNode[] = []
  for (const node of nodes.values()) {
    const parent = node.parent_id ? nodes.get(node.parent_id) : undefined
    if (parent) parent.children.push(node)
    else roots.push(node)
  }
  return roots
}

/** Ancestor ids of `versionId` (its branch), nearest first excludes itself. */
function branchOf(versions: DatasetVersion[], versionId: string | null): Set<string> {
  const byId = new Map(versions.map((v) => [v.id, v]))
  const chain = new Set<string>()
  let current = versionId ? byId.get(versionId) : undefined
  while (current) {
    chain.add(current.id)
    current = current.parent_id ? byId.get(current.parent_id) : undefined
  }
  return chain
}

const ORIGIN_VARIANTS: Record<string, string> = {
  uploaded: 'text-sky-400 border-sky-400/40',
  generated: 'text-violet-400 border-violet-400/40',
  labeled: 'text-amber-400 border-amber-400/40',
  cleaned: 'text-emerald-400 border-emerald-400/40',
}

export function VersionTree({
  versions,
  selectedId,
  onSelect,
}: {
  versions: DatasetVersion[]
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  const tree = buildTree(versions)
  const branch = branchOf(versions, selectedId)

  const renderNode = (node: TreeNode, depth: number) => {
    const onBranch = branch.has(node.id)
    const selected = node.id === selectedId
    return (
      <motion.li
        key={node.id}
        layout
        initial={{ opacity: 0, y: -6 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative"
        style={{ paddingLeft: depth * 18 }}
      >
        <button
          type="button"
          onClick={() => onSelect(node.id)}
          aria-current={selected ? 'true' : undefined}
          data-branch={onBranch && !selected ? 'ancestor' : undefined}
          className={cn(
            'w-full rounded-md border border-transparent px-2.5 py-2 text-left transition-colors hover:bg-muted/60',
            onBranch && !selected && 'bg-muted/40',
            selected && 'border-primary/50 bg-primary/10',
          )}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-2 text-sm font-medium">
              v{node.number}
              <Badge variant="outline" className={cn('capitalize', ORIGIN_VARIANTS[node.origin])}>
                {node.origin}
              </Badge>
            </span>
            <span className="text-xs text-muted-foreground">
              {fmtNumber(node.row_count)} · {timeAgo(node.created_at)}
            </span>
          </div>
          <ProvenanceBar summary={node.provenance_summary ?? {}} className="mt-1.5" />
        </button>
        {depth > 0 && (
          <span aria-hidden className="absolute top-0 h-full border-l border-border" style={{ left: (depth - 1) * 18 + 9 }} />
        )}
        {node.children.length > 0 && (
          <ul className="flex flex-col gap-1">
            {node.children.map((child) => renderNode(child, depth + 1))}
          </ul>
        )}
      </motion.li>
    )
  }

  return (
    <ul className="flex flex-col gap-1">
      {tree.map((root) => renderNode(root, 0))}
      {tree.length === 0 && (
        <li className="px-2 py-6 text-center text-sm text-muted-foreground">
          No Dataset Versions yet — upload a file to start the tree.
        </li>
      )}
    </ul>
  )
}
